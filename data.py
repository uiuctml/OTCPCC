from torch.utils.data import DataLoader
from typing import *

from hierarchy.data import get_k_shot
import dataset.cifar100.data as cifar100
import dataset.cifar10.data as cifar10
import dataset.breeds2.data as breeds
import dataset.inat.data as inat


def make_kshot_loader(num_workers : int, batch_size : int, k : int, layer : str, 
                      seed : int, dataset : str, breeds_setting : str) -> Tuple[DataLoader, DataLoader]:
    '''
    Prepare one-shot train loader and full test loader. In train dataset, we have k
    image for each class (of indicated layer) in train set.
    Only call this function on unseen **target** train/test set.
    Args:
        batch_size : batch size of train/test loader
        k : k shot on train set
        layer : on which layer to sample k image per class
        task : in / sub
    '''
    # we use one shot setting only for target hierarchy
    # where we have to fine tune because of distribution shift at fine level
    train_dataloader, test_dataloader = make_dataloader(num_workers, batch_size, dataset, 'tt', breeds_setting)
    train_subset, _ = get_k_shot(k, train_dataloader.dataset, layer, seed)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                                pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataloader.dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=num_workers)
    dataloaders = (train_loader, test_loader)
    return dataloaders

def make_dataloader(num_workers : int, batch_size : int, dataset : str, 
                    task : str, breeds_setting : str = None) -> Tuple[DataLoader, DataLoader]:
    '''
    Creat (a subset of) train test dataloader. Train & test has the same number of classes.
    Args:
        num_workers : number of workers of train and test loader.
        batch_size : batch size of train and test loader.
        dataset : MNIST | CIFAR | BREEDS
        task : ss | st | tt
        breeds_setting : living17 | entity13 | entity30 | nonliving26
    '''
    if dataset == 'CIFAR100':
        train_loader, test_loader = cifar100.make_dataloader(num_workers, batch_size, task)
    elif dataset == 'CIFAR10':
        train_loader, test_loader = cifar10.make_dataloader(num_workers, batch_size, task)
    elif dataset == 'BREEDS':
        if breeds_setting is None:
            raise ValueError("Please specify --breeds_setting as any of living17 | entity13 | entity30 | nonliving26")
        train_loader, test_loader = breeds.make_dataloader(num_workers, batch_size, task, breeds_setting)
    elif dataset == 'INAT':
        train_loader, test_loader = inat.make_dataloader(num_workers, batch_size, task)
    return train_loader, test_loader
