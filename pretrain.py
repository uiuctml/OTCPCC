import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

import re
import numpy as np
from tqdm import tqdm

from datetime import datetime
import argparse
import os
import json
from typing import *

from model import init_model
from data import make_dataloader
from loss import CPCCLoss
from param import init_optim_schedule, load_params
from utils import get_layer_prob_from_fine, seed_everything

def train_source(train_loader : DataLoader, test_loader : DataLoader, 
                       device : torch.device, 
                       save_dir : str, seed : int, epochs : int,
                       dataset_name : str, breeds_setting : str, hyper) -> None:
    dataset = train_loader.dataset
    num_classes = [len(dataset.fine_names)]

    optim_config, scheduler_config = hyper['optimizer'], hyper['scheduler']
    init_config = {"dataset":dataset_name,
                "batch_size":train_loader.batch_size,
                "epochs":epochs,
                "seed":seed, # will be replaced until we reach the max seeds
                "save_dir":save_dir,
                "_num_workers":train_loader.num_workers,
                "lamb":lamb,
                "breeds_setting":breeds_setting,
                'metric':metric,
                }
    
    config = {**init_config, **optim_config, **scheduler_config}    

    criterion_ce = nn.CrossEntropyLoss()
        
    if metric:
        criterion_cpcc = CPCCLoss(dataset, metric, height, emd_weights, dist_weights)
    else:
        criterion_cpcc = None
    
    if rank <= 0:
        with open(save_dir+'/config.json', 'w') as fp:
            json.dump(config, fp, sort_keys=True, indent=4)
        wandb.init(project=f"{dataset_name}_onestage_pretrain", 
                entity="structured_task",
                name=datetime.now().strftime("%m%d%Y%H%M%S"),
                config=config,
                settings=wandb.Settings(code_dir="."))

    if 'WORLD_SIZE' in os.environ and world_size > 0:
        dist.init_process_group()
    torch.set_float32_matmul_precision('high')
    
    out_dir = save_dir+f"/seed{seed}.pth"
    model = init_model(dataset_name, num_classes, device, arch)
    optimizer, scheduler = init_optim_schedule(model, hyper)

    # Check if a checkpoint exists at the start
    checkpoint_filepath = ''
    start_epoch = 0
    max_epoch_num = -1
    regex = re.compile(r'\d+')

    for file in os.listdir(checkpoint_dir):
        epoch_num = int(regex.findall(file)[0])
        if epoch_num > max_epoch_num:
            max_epoch_num = epoch_num
            checkpoint_filepath = os.path.join(checkpoint_dir, file)
    if max_epoch_num != -1:
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = checkpoint['scheduler']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    epochs_durations = []

    for epoch in range(start_epoch, epochs):
        if rank > 0:
            train_loader.sampler.set_epoch(epoch)
        t_start = datetime.now() # record the time for each epoch
        model.train()
        train_fine_accs = []
        train_coarse_accs = []
        train_losses_ce = []
        train_losses_cpcc = []
        
        for idx, (data, _, target_coarse, _, target_fine) in enumerate(train_loader):
            data = data.to(device)
            target_fine = target_fine.to(device)
            target_coarse = target_coarse.to(device)
            optimizer.zero_grad()

            representation, output_fine = model(data)
            loss_ce = criterion_ce(output_fine, target_fine)
            
            if metric:
                loss_cpcc = lamb * criterion_cpcc(representation, target_fine)
                loss = loss_ce + loss_cpcc
            else:
                loss = loss_ce
            
            loss.backward()
            optimizer.step()

            if metric:
                train_losses_cpcc.append(loss_cpcc.detach())
            train_losses_ce.append(loss_ce.detach())
        
            prob_fine = F.softmax(output_fine,dim=1)
            pred_fine = prob_fine.argmax(dim=1)
            acc_fine = pred_fine.eq(target_fine).flatten().tolist()
            train_fine_accs.extend(acc_fine)

            prob_coarse = get_layer_prob_from_fine(prob_fine, dataset.coarse_map)
            pred_coarse = prob_coarse.argmax(dim=1)
            acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
            train_coarse_accs.extend(acc_coarse)

            if idx % 100 == 1 and rank <= 0:
                if not metric:
                    loss_cpcc = -1
                print(f"Train Loss: {loss}, Acc_fine: {sum(train_fine_accs)/len(train_fine_accs)}, Acc_coarse: {sum(train_coarse_accs)/len(train_coarse_accs)}, loss_cpcc: {loss_cpcc}")
        
        scheduler.step()
        
        model.eval() 
        test_fine_accs = []
        test_coarse_accs = []
        test_losses_ce = []
        test_losses_cpcc = []
        
        with torch.no_grad():
            for idx, (data, _, target_coarse, _, target_fine) in enumerate(test_loader):
                data = data.to(device)
                target_coarse = target_coarse.to(device)
                target_fine = target_fine.to(device)

                representation, output_fine = model(data)
                loss_ce = criterion_ce(output_fine, target_fine)
                
                if metric:
                    loss_cpcc = lamb * criterion_cpcc(representation, target_fine)
                    loss = loss_ce + loss_cpcc
                    test_losses_cpcc.append(loss_cpcc.detach())
                else:
                    loss = loss_ce
                test_losses_ce.append(loss_ce.detach())

                prob_fine = F.softmax(output_fine,dim=1)
                pred_fine = prob_fine.argmax(dim=1)
                acc_fine = pred_fine.eq(target_fine).flatten().tolist()
                test_fine_accs.extend(acc_fine)

                prob_coarse = get_layer_prob_from_fine(prob_fine, dataset.coarse_map)
                pred_coarse = prob_coarse.argmax(dim=1)
                acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                test_coarse_accs.extend(acc_coarse)

        
        t_end = datetime.now()
        t_delta = (t_end-t_start).total_seconds()
        
        if rank <= 0:
            print(f"Val loss_ce: {sum(test_losses_ce)/len(test_losses_ce)}, Acc_fine: {sum(test_fine_accs)/len(test_fine_accs)}, Acc_coarse: {sum(train_coarse_accs)/len(train_coarse_accs)}")
            print(f"Epoch {epoch} takes {t_delta} sec.")
            epochs_durations.append(t_delta)

            log_dict = {"train_fine_acc":sum(train_fine_accs)/len(train_fine_accs), 
                        "train_coarse_acc":sum(train_coarse_accs)/len(train_coarse_accs),
                        "train_losses_ce":(sum(train_losses_ce)/len(train_losses_ce)).item(),
                        "val_fine_acc":sum(test_fine_accs)/len(test_fine_accs),
                        "val_losses_ce":(sum(test_losses_ce)/len(test_losses_ce)).item(),
                        "val_coarse_acc":sum(test_coarse_accs)/len(test_coarse_accs),
                        "per_epoch_seconds" : t_delta
                    }
            if metric: # batch-CPCC
                log_dict["train_losses_cpcc"] = (sum(train_losses_cpcc)/len(train_losses_cpcc)).item()
                log_dict["val_losses_cpcc"] = (sum(test_losses_cpcc)/len(test_losses_cpcc)).item()
            else:
                log_dict["train_losses_cpcc"] = -1
                log_dict["val_losses_cpcc"] = -1
            wandb.log(log_dict)

            # Save the model every certain number of epochs
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                save_info = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler' : scheduler,
                    'train_fine_acc': log_dict['train_fine_acc'],
                    'train_coarse_acc' : log_dict['train_coarse_acc'],
                    'train_losses_ce': log_dict['train_losses_ce'],
                    "val_fine_acc" : log_dict['val_fine_acc'],
                    'val_coarse_acc' : log_dict['val_coarse_acc'],
                    "val_losses_ce" : log_dict['val_losses_ce'],
                    "train_losses_cpcc": log_dict["train_losses_cpcc"],
                    "val_losses_cpcc" : log_dict["val_losses_cpcc"],
                    "per_epoch_seconds" : t_delta
                    }
                checkpoint_filepath = os.path.join(checkpoint_dir, f"checkpoint_{epoch}_{seed}.pth")
                torch.save(save_info, checkpoint_filepath)
                if epoch == epochs - 1:
                    torch.save(model.state_dict(), out_dir)
                    with open(save_dir + '/pretrain_metrics.json', "w") as f:
                        json.dump(log_dict, f)
    

    wandb.finish()

    if rank <= 0:
        avg_epoch_duration = sum(epochs_durations) / len(epochs_durations)
        with open(save_dir+f'/per_epoch_time_{seed}.txt', 'w') as f:
            f.write(f"Average epoch duration: {avg_epoch_duration} sec.\n")

    ddp_cleanup()
    return

def eval_source(train_loader, test_loader, 
                              save_dir : str, seeds : int, device : torch.device, 
                              level : str, dataset_name : str):

    dataset = train_loader.dataset

    metrics = {
        "train_top1": [],
        "train_top2": [],
        "train_losses": [],
        "val_top1": [],
        "val_top2": [],
        "val_losses": []
    }

    criterion = torch.nn.CrossEntropyLoss().to(device)

    for seed in range(seeds):
        model_path = save_dir + f"/seed{seed}.pth"
        num_classes = len(dataset.fine_names)
        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)
        criterion = nn.CrossEntropyLoss()

        model = init_model(dataset_name, [num_classes], device, arch)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        train_top1, train_top2, test_top1, test_top2 = 0, 0, 0, 0
        train_losses, test_losses = [], []

        with torch.no_grad():
            for data, target_coarsest, target_coarse, target_mid, target_fine in tqdm(train_loader, desc=f'{level}_train_loader'):
                data = data.to(device)
                target_coarse = target_coarse.to(device)
                target_fine = target_fine.to(device)
                target_mid = target_mid.to(device)
                target_coarsest = target_coarsest.to(device)
                
                target = target_fine

                _, output = model(data)
                loss = criterion(output, target)
                train_losses.append(loss.item())

                prob = F.softmax(output, dim=1)
                if level == 'fine':
                    pred1 = prob.argmax(dim=1, keepdim=False)
                    train_top1 += pred1.eq(target).sum().item()

                    pred2 = (prob.topk(k=2, dim=1)[1]).T
                    target_reshaped = target.unsqueeze(0).expand_as(pred2)
                    train_top2 += pred2.eq(target_reshaped).sum().item()
                elif level == 'coarse':
                    coarse_targets_map = dataset.coarse_map
                    prob_coarse = get_layer_prob_from_fine(prob, coarse_targets_map)
                    pred_coarse = prob_coarse.argmax(dim=1)
                    train_top1 += pred_coarse.eq(target_coarse).sum().item()

                

        with torch.no_grad():
            for data, target_coarsest, target_coarse, target_mid, target_fine in tqdm(test_loader,desc=f'{level}_test_loader'):
                data = data.to(device)
                target_coarse = target_coarse.to(device)
                target_fine = target_fine.to(device)
                target_mid = target_mid.to(device)
                target_coarsest = target_coarsest.to(device)

                target = target_fine

                _, output = model(data)
                loss = criterion(output, target)
                test_losses.append(loss.item())
                prob = F.softmax(output, dim=1)

                if level == 'fine':
                    pred1 = prob.argmax(dim=1, keepdim=False)
                    test_top1 += pred1.eq(target).sum().item()
                    pred2 = (prob.topk(k=2, dim=1)[1]).T
                    target_reshaped = target.unsqueeze(0).expand_as(pred2)
                    test_top2 += pred2.eq(target_reshaped).sum().item()
                elif level == 'coarse':
                    coarse_targets_map = dataset.coarse_map
                    prob_coarse = get_layer_prob_from_fine(prob, coarse_targets_map)
                    pred_coarse = prob_coarse.argmax(dim=1)
                    test_top1 += pred_coarse.eq(target_coarse).sum().item()

        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)
        metrics["train_top1"].append(train_top1/train_size)
        metrics["train_top2"].append(train_top2/train_size)
        metrics["train_losses"].append(np.mean(train_losses))
        metrics["val_top1"].append(test_top1/test_size)
        metrics["val_top2"].append(test_top2/test_size)
        metrics["val_losses"].append(np.mean(test_losses))

    metrics_summary = {
        key: {
            "values": values,
            "mean": np.mean(values),
            "std": np.std(values)
        } for key, values in metrics.items()
    }
    print(metrics_summary)
    with open(save_dir + f"/{level}_source_classification.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)


def main(train):
    for seed in range(seeds):
        seed_everything(seed)
        hyper = load_params(dataset_name, 'pre', breeds_setting = breeds_setting) # pretrain
        epochs = hyper['epochs']
        train_loader, test_loader = make_dataloader(num_workers, batch_size, dataset_name, 'ss', breeds_setting)
        if train:
            train_source(train_loader, test_loader, device, save_dir, seed, epochs, dataset_name, breeds_setting, hyper)
        else:
            for level in ['fine','coarse']:
                eval_source(train_loader, test_loader, save_dir, seeds, device, level, dataset_name)
    return


def ddp_setup():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group()

    device_id = rank
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device_id) 

    return device, device_id, rank, world_size

def ddp_cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./exp", type=str, help='directory that you want to save your experiment results')
    parser.add_argument("--timestamp", required=True, help=r'your unique experiment id, hint: datetime.now().strftime("%m%d%Y%H%M%S")') 
    
    parser.add_argument("--dataset", required=True, help='CIFAR10/CIFAR100/BREEDS/INAT')
    parser.add_argument("--breeds_setting", default="", type=str, help='living17, nonliving26, entity13, entity30')
    parser.add_argument("--save_freq", type=int, default=20)
    
    parser.add_argument("--cpcc", required=True, type=str, nargs='?', const='', help='distance metric in CPCC, emd/sk/swd/fft/l2')
    parser.add_argument("--lamb",type=float,default=1,help='strength of CPCC regularization')
    
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seeds", type=int, default=5)    

    parser.add_argument("--local-rank", type=int, default=0) # DDP
    parser.add_argument("--train", required=True, type=int)

    parser.add_argument("--arch", type=str, default='res18', help='res18 | res34 | res50 | vitb16 | vitl16')

    parser.add_argument("--height", type=int, default=2, help='CPCC tree height, can be either 2, 3, or 4')
    parser.add_argument("--emd_weights", type=int, default=0, help='0 = uniform, 1 = dist, 2 = inv-dist')
    parser.add_argument("--dist_weights", type=int, default=1, help='distance of the top left diagonal block in CIFAR100')


    if 'RANK' not in os.environ:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = -1
    
    # for DDP
    else:
        local_rank = int(os.getenv('LOCAL_RANK', 0)) 
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print("device: ", device)
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

    args = parser.parse_args()
    arch = args.arch
    height = args.height
    emd_weights = args.emd_weights
    dist_weights = args.dist_weights


    timestamp = args.timestamp
    dataset_name = args.dataset
    metric = args.cpcc
    
    num_workers = args.num_workers
    batch_size = args.batch_size
    save_freq = args.save_freq
    seeds = args.seeds
    lamb = args.lamb

    root = args.root 
    
    root = f'{root}/hierarchy_results/{dataset_name}' 
    save_dir = root + '/' + timestamp 
    if not os.path.exists(save_dir) and rank <= 0:
        os.makedirs(save_dir)

    breeds_setting = args.breeds_setting
    if breeds_setting:
        checkpoint_dir = save_dir + f'/checkpoint/{breeds_setting}' 
    else:
        checkpoint_dir = save_dir + '/checkpoint'
    if not os.path.exists(checkpoint_dir) and rank <= 0:
        os.makedirs(checkpoint_dir)

    main(args.train)
