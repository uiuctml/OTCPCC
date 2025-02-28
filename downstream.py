import json

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle

from datetime import datetime
import argparse
import os
from typing import *

from model import init_model
from data import make_kshot_loader, make_dataloader
from param import init_optim_schedule, load_params
from utils import seed_everything, get_layer_prob_from_fine

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, average_precision_score

from loss import CPCCLoss

def transfer_target(save_dir : str, seed : int, device : torch.device, 
                    batch_size : int, level : str, num_workers : int, 
                    dataset_name : str, breeds_setting : str,
                    hyper, epochs) -> nn.Module:
    '''
        Transfer to target sets on new level.
    '''
    out_dir = save_dir+f"/{level}_seed{seed}.pth"

    if os.path.exists(out_dir):
        print("One shot target transfer skipped.")
        return

    train_loader, test_loader = make_kshot_loader(num_workers, batch_size, 1, level,
                                                seed, dataset_name, breeds_setting) # we use one shot on train set, tt dataloader
    dataset = train_loader.dataset.dataset # loader contains Subset
    num_classes = len(dataset.fine_names)
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    criterion = nn.CrossEntropyLoss() # no CPCC in downstream task
    
    model = init_model(dataset_name, [num_classes], device)
    
    model_dict = model.state_dict()
    # load pretrained seed 0, call this function 
    trained_dict = {k: v for k, v in torch.load(save_dir+"/seed0.pth").items() if (k in model_dict) and ("fc" not in k)}
    model_dict.update(trained_dict) 
    model.load_state_dict(model_dict)

    for param in model.parameters(): # Freeze Encoder, fit last linear layer
        param.requires_grad = False

    model._orig_mod.fc = nn.Linear(model.out_features, num_classes).to(device)

    init_config = {"_batch_size":train_loader.batch_size,
                    "epochs":epochs,
                    "seed":seed,
                    "save_dir":save_dir,
                    "_num_workers":train_loader.num_workers,
                    "breeds_setting":breeds_setting,
                    "level":level
                    }
    optim_config, scheduler_config = hyper['optimizer'], hyper['scheduler']
    config = {**init_config, **optim_config, **scheduler_config}  

    wandb.init(project=f"{dataset_name}-oneshot",
            entity="structured_task",
            name=datetime.now().strftime("%m%d%Y%H%M%S"),
            config=config,
            settings=wandb.Settings(code_dir=".")
    )
   
    optimizer, scheduler = init_optim_schedule(model, hyper)

    for epoch in range(epochs):
        t_start = datetime.now()
        model.eval()
        test_top1 = 0
        test_top2 = 0
        test_losses = []
       
        with torch.no_grad():
            for (data, _, _, target_mid, target_fine) in test_loader:
                data = data.to(device)
                if level == 'fine':
                    target_fine = target_fine.to(device)
                    target = target_fine
                elif level == 'mid':
                    target_mid = target_mid.to(device)
                    target = target_mid

                _, output = model(data)
                loss = criterion(output, target)
                test_losses.append(loss)

                prob = F.softmax(output,dim=1)

                # top 1
                pred1 = prob.argmax(dim=1, keepdim=False) 
                top1_correct = pred1.eq(target).sum()
                test_top1 += top1_correct

                # top 2
                pred2 = (prob.topk(k=2, dim=1)[1]).T # 5 * batch_size
                target_reshaped = target.unsqueeze(0).expand_as(pred2)
                top2_correct = pred2.eq(target_reshaped).sum() 
                test_top2 += top2_correct
        
        print(f"Val loss: {sum(test_losses)/len(test_losses)}, Top1_{level}: {test_top1/test_size} "
              f"Top2_{level} : {test_top2/test_size}")

        model.train()
        train_top1 = 0
        train_top2 = 0
        train_losses = []
        for idx, (data, _, _, target_mid, target_fine) in enumerate(train_loader):
            data = data.to(device)
            if level == 'fine':
                target_fine = target_fine.to(device)
                target = target_fine
            elif level == 'mid':
                target_mid = target_mid.to(device)
                target = target_mid

            optimizer.zero_grad()
            _, output = model(data)
            loss = criterion(output, target)
            train_losses.append(loss)
    
            loss.backward()
            optimizer.step()
            
            prob = F.softmax(output,dim=1)

            pred1 = prob.argmax(dim=1, keepdim=False) 
            top1_correct = pred1.eq(target).sum()
            train_top1 += top1_correct

            pred2 = (prob.topk(k=2, dim=1)[1]).T 
            target_reshaped = target.unsqueeze(0).expand_as(pred2)
            top2_correct = pred2.eq(target_reshaped).sum() 
            train_top2 += top2_correct
            
            if idx % 100 == 0:
                print(f"Train loss: {sum(train_losses)/len(train_losses)}, Top1_{level}: {train_top1/train_size} "
                      f"Top2_{level} : {train_top2/train_size}")

        scheduler.step()
        
        t_end = datetime.now()
        t_delta = (t_end-t_start).total_seconds()
        print(f"Epoch {epoch} takes {t_delta} sec.")
        
        wandb.log({
            "train_top1":train_top1/train_size,
            "train_top2":train_top2/train_size,
            "train_losses":sum(train_losses)/len(train_losses),
            "val_top1":test_top1/test_size,
            "val_top2":test_top2/test_size,
            "val_losses":sum(test_losses)/len(test_losses),
        })
    
    torch.save(model.state_dict(), save_dir+f"/{level}_seed{seed}.pth")
    wandb.finish()
    
    return model

def eval_target(save_dir : str, seeds : int, device : torch.device, 
                batch_size : int, level : str, num_workers : int, 
                dataset_name : str, breeds_setting : str):

    if os.path.exists(save_dir+f'/{level}_target_classification.json'):
        print(f'{level} target classification evaluation skipped.')
        return
    
    source_test_loader = make_dataloader(num_workers, batch_size, dataset_name, 'ss', breeds_setting)[1]
    
    if 'level' == 'fine':
        train_loader, test_loader = make_kshot_loader(num_workers, batch_size, 1, level, 
                                                    seeds, dataset_name, breeds_setting)
    else:
        train_loader, test_loader = make_dataloader(num_workers, batch_size, dataset_name, 'st', breeds_setting)


    metrics = {
        "train_top1": [],
        "train_top2": [],
        "train_losses": [],
        "val_top1": [],
        "val_top2": [],
        "val_losses": []
    }

    criterion = torch.nn.CrossEntropyLoss()

    for seed in range(seeds):
        if level == 'fine':
            model_path = save_dir + f"/{level}_seed{seed}.pth" # one-shot
        elif level == 'coarse':
            model_path = save_dir+f"/seed{seed}.pth" # zero-shot

        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)
        criterion = nn.CrossEntropyLoss() # no CPCC in downstream task
        
        if level == 'fine':
            model = init_model(dataset_name, [len(test_loader.dataset.fine_names)], device)
        elif level == 'coarse':
            model = init_model(dataset_name, [len(source_test_loader.dataset.fine_names)], device)
       
        new_state_dict = {}
        for key, value in torch.load(model_path).items():
            new_key = key.replace(".module", "")
            
            if new_key == '_orig_mod.fc2.weight':
                new_state_dict['_orig_mod.fc.weight'] = value
                del new_state_dict['_orig_mod.fc1.weight']
            elif new_key == '_orig_mod.fc2.bias':
                new_state_dict['_orig_mod.fc.bias'] = value
                del new_state_dict['_orig_mod.fc1.bias']
            else:
                new_state_dict[new_key] = value
        
        
        model.load_state_dict(new_state_dict)
       
        model.eval()

        train_top1, train_top2, test_top1, test_top2 = 0, 0, 0, 0
        train_losses, test_losses = [], []

        with torch.no_grad():

            for data, target_coarsest, target_coarse, target_mid, target_fine in tqdm(test_loader):
                data = data.to(device)
                target_coarse = target_coarse.to(device)
                target_fine = target_fine.to(device)
                target_mid = target_mid.to(device)
                target_coarsest = target_coarsest.to(device)

                
                if level == 'coarse':
                    target = target_coarse
                elif level == 'fine':
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
                    coarse_targets_map = source_test_loader.dataset.coarse_map
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
    with open(save_dir + f"/{level}_target_classification.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

def retrieval(seeds, save_dir, task_name, train_loader, 
             test_loader, device, dataset_name, levels):
    """
    Function to perform image retrieval and calculate similarity at a specified level (coarse or fine).

    Parameters:
    - seeds: Number of seeds.
    - save_dir: Directory to save output.
    - task_name: source or train.
    - train_loader: Training data loader.
    - test_loader: Testing data loader.
    - exp_name: Experiment name.
    - device: Computation device.
    - dataset_name: Name of the dataset.
    - levels: Levels of retrieval (in ['coarse', 'mid', 'fine']).
    """
    
    # Initialization
    train_dataset = train_loader.dataset

    # Loop over seeds
    for level in levels:
        if os.path.exists(save_dir+f'/{level}_{task_name}_retrieval.json'):
            print('retrieval skipped.')
            return
        mAPs = []
        precisions = []
        recalls = []
        
        for seed in range(seeds):
            # Load model based on level
            if level == 'coarsest':
                target_num = len(train_dataset.coarsest_names)
            elif level == 'coarse':
                target_num = len(train_dataset.coarse_names)
            elif level == 'mid':
                target_num = len(train_dataset.mid_names)
            else:
                assert level == 'fine'
                target_num = len(train_dataset.fine_names)

            model = init_model(dataset_name, [len(train_dataset.fine_names)], device)
                        
            new_state_dict = {}
            for key, value in torch.load(save_dir+f"/seed{seed}.pth").items():
                new_key = key.replace(".module", "")
                
                if new_key == '_orig_mod.fc2.weight':
                    new_state_dict['_orig_mod.fc.weight'] = value
                    del new_state_dict['_orig_mod.fc1.weight']
                elif new_key == '_orig_mod.fc2.bias':
                    new_state_dict['_orig_mod.fc.bias'] = value
                    del new_state_dict['_orig_mod.fc1.bias']
                else:
                    new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)

            model.eval()

            # Initialize prototypes and other variables
            aps = []
            prototypes = {i: [] for i in range(target_num)}
            test_scores = {i: [] for i in range(target_num)}
            test_truth = {i: [] for i in range(target_num)}
            all_targets = []
            retrieval_results = []

            # Process train_loader to create prototypes
            with torch.no_grad():
                for item in train_loader:
                    data = item[0].to(device)
                    if level ==  'coarsest':
                        target = item[-4]
                    elif level ==  'coarse':
                        target = item[-3]
                    elif level == 'mid':
                        target = item[-2]
                    else:
                        assert level == 'fine'
                        target = item[-1]
                    representation, _ = model(data)
                    for it, t in enumerate(target):
                        prototypes[t.item()].append(representation[it].cpu().numpy())

                # Calculate mean of prototypes
                for i in range(target_num):
                    prototypes[i] = np.mean(np.stack(prototypes[i], axis=0), axis=0)


                # Process test_loader for retrieval
                for item in tqdm(test_loader):
                    data = item[0].to(device)
                    if level ==  'coarsest':
                        target = item[-4]
                    elif level == 'coarse':
                        target = item[-3]
                    elif level == 'mid':
                        target = item[-2]
                    else:
                        assert level == 'fine'
                        target = item[-1]
                    all_targets.extend(target.cpu().numpy())
                    test_embs, _ = model(data)
                    test_embs = test_embs.cpu().detach().numpy()
                    
                    for it, t in enumerate(target):
                        test_emb = test_embs[it]
                        similarities = []
                        for i in range(target_num):
                            similarity = np.dot(test_emb, prototypes[i])/(np.linalg.norm(test_emb)*np.linalg.norm(prototypes[i]))
                            similarities.append(similarity)
                            test_scores[i].append(np.dot(test_emb, prototypes[i])/(np.linalg.norm(test_emb)*np.linalg.norm(prototypes[i])))
                            test_truth[i].append(int(t.item() == i))
                        retrieval_results.append(np.argmax(similarities))

                # Compute metrics
                for i in range(target_num):
                    aps.append(average_precision_score(test_truth[i], test_scores[i]))

            # Compute average metrics
            mAP = np.mean(aps)
            precision = precision_score(all_targets, retrieval_results, average='macro')
            recall = recall_score(all_targets, retrieval_results, average='macro')

            # Append results for each seed
            mAPs.append(mAP)
            precisions.append(precision)
            recalls.append(recall)

        # Output results
        out = {
            'mAP': mAPs,
            'mean_mAP': np.mean(mAPs),
            'std_mAP': np.std(mAPs),
            'precisions': precisions,
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'recalls': recalls,
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls)
        }

        # Save results to file
        save_filename = f'/{level}_{task_name}_retrieval.json'
        with open(save_dir + save_filename, 'w') as fp:
            json.dump(out, fp, indent=4)

    return 

def feature_extractor(dataloader, seed, device):
    dataset = dataloader.dataset
    dataset_name = dataset.dataset_name
    model = init_model(dataset_name, [len(dataset.fine_names)], device)    

    new_state_dict = {}
    for key, value in torch.load(save_dir+f"/seed{seed}.pth").items():
        new_key = key.replace(".module", "")
        
        if new_key == '_orig_mod.fc2.weight':
            new_state_dict['_orig_mod.fc.weight'] = value
            del new_state_dict['_orig_mod.fc1.weight']
        elif new_key == '_orig_mod.fc2.bias':
            new_state_dict['_orig_mod.fc.bias'] = value
            del new_state_dict['_orig_mod.fc1.bias']
        else:
            new_state_dict[new_key] = value    
    model.load_state_dict(new_state_dict)

    features = []
    probs = []
    targets_one = []
    targets_coarse = []
    model.eval()
    with torch.no_grad():
        for item in dataloader:
            data = item[0]
            target_one = item[-1] # add fine target
            data = data.to(device)
            target_one = target_one.to(device)
            feature, output = model(data)
            prob_one = F.softmax(output,dim=1)
            probs.append(prob_one.cpu().detach().numpy())
            features.append(feature.cpu().detach().numpy())
            targets_one.append(target_one.cpu().detach().numpy())
            if len(item) == 5:
                target_coarse = item[2]
                target_coarse = target_coarse.to(device)
                targets_coarse.append(target_coarse.cpu().detach().numpy())
    features = np.concatenate(features,axis=0)
    targets_one = np.concatenate(targets_one,axis=0)
    probs = np.concatenate(probs,axis=0)
    if len(targets_coarse) > 0:
        targets_coarse = np.concatenate(targets_coarse,axis=0)  
    
    return (features, probs, targets_one, targets_coarse)


def fullCPCC(data_loader, task_name, metric, seeds):

    if os.path.exists(save_dir+f'/{task_name}_CPCC.json'):
        print('CPCC skipped.')
        return
    
    all_cpcc = []
    for seed in range(seeds):
        if os.path.exists(save_dir + f'/test_features_seed{seed}.pkl') and os.path.exists(save_dir + f'/test_labels_seed{seed}.pkl'):
            with open(save_dir + f'/test_features_seed{seed}.pkl', 'rb') as fp:
                features = pickle.load(fp)
            with open(save_dir + f'/test_labels_seed{seed}.pkl', 'rb') as fp:
                targets_fine = pickle.load(fp)
        else:
            features, _, targets_fine, _ = feature_extractor(data_loader, seed, device)
            pickle.dump(features, open(save_dir + f'/test_features_seed{seed}.pkl', 'wb'))
            pickle.dump(targets_fine, open(save_dir + f'/test_labels_seed{seed}.pkl', 'wb'))
            
        criterion = CPCCLoss(data_loader.dataset, metric, emd_weights=emd_weights, height=height, dist_weights = dist_weights)
        cpcc = 1 - criterion(torch.tensor(features), torch.tensor(targets_fine)).item()
        all_cpcc.append(cpcc)
    
    result = {'cpcc': (np.mean(all_cpcc), np.std(all_cpcc), all_cpcc)}
    
    with open(save_dir+f'/{task_name}_CPCC.json', 'w') as fp:
        json.dump(result, fp, indent=4)

    return result

def main():
    
    # Train
    for seed in range(seeds):
        seed_everything(seed)
        
        for level in ['fine']: 
            hyper = load_params(dataset_name, 'down', level, breeds_setting)
            epochs = hyper['epochs']
            transfer_target(save_dir, seed, device, batch_size, level, num_workers, dataset_name, breeds_setting, hyper, epochs)

    for level in ['fine','coarse']:  
        eval_target(save_dir, seeds, device, batch_size, level, num_workers, dataset_name, breeds_setting)

    # source-source
    task_name = 'ss'
    train_loader, test_loader = make_dataloader(num_workers, batch_size, dataset_name, task_name, breeds_setting)
    retrieval(seeds, save_dir, task_name, train_loader, test_loader, device, dataset_name, ['coarse','fine'])
    fullCPCC(test_loader, task_name, cpcc, seeds)

    # source-target
    task_name = 'st'
    train_loader, test_loader = make_dataloader(num_workers, batch_size, dataset_name, task_name, breeds_setting)
    retrieval(seeds, save_dir, task_name, train_loader, test_loader, device, dataset_name, ['coarse'])

    task_name = 'full'
    train_loader, test_loader = make_dataloader(num_workers, batch_size, dataset_name, task_name, breeds_setting)
    fullCPCC(test_loader, task_name, cpcc, seeds)
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./exp", type=str, help='directory that you want to save your experiment results')
    parser.add_argument("--timestamp", required=True, help=r'your unique experiment id, hint: datetime.now().strftime("%m%d%Y%H%M%S")') 
    parser.add_argument("--dataset", required=True, help='CIFAR10/CIFAR100/BREEDS/INAT')
    parser.add_argument("--breeds_setting", default="", type=str, help='living17, nonliving26, entity13, entity30')
    parser.add_argument("--cpcc", type=str, nargs='?', const='', help='distance metric in CPCC, emd/sk/swd/fft/l2')
    
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seeds", type=int,default=1)

    parser.add_argument("--height", type=int, default=2, help='CPCC tree height, can be either 2, 3, or 4')
    parser.add_argument("--emd_weights", type=int, default=0, help='0 = uniform, 1 = dist, 2 = inv-dist')
    parser.add_argument("--dist_weights", type=int, default=1, help='distance of the top left diagonal block in CIFAR100')


    args = parser.parse_args()
    timestamp = args.timestamp
    dataset_name = args.dataset
    height = args.height
    emd_weights = args.emd_weights
    dist_weights = args.dist_weights

    num_workers = args.num_workers
    batch_size = args.batch_size
    seeds = args.seeds
    cpcc = args.cpcc

    root = args.root 
    
    root = f'{root}/hierarchy_results/{dataset_name}' 
    save_dir = root + '/' + timestamp 
    
    breeds_setting = args.breeds_setting
    if breeds_setting:
        checkpoint_dir = save_dir + f'/checkpoint/{breeds_setting}' 
    else:
        checkpoint_dir = save_dir + '/checkpoint'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()