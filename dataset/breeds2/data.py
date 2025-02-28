from torch.utils.data import DataLoader, Subset

from robustness.tools.folder import ImageFolder
from robustness.datasets import CustomImageNet
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26

import os
from typing import *
import numpy as np
import json

from hierarchy.data import Hierarchy
from torch.utils.data.distributed import DistributedSampler


class HierarchyBREEDS2(ImageFolder, Hierarchy):
    def __init__(self, info_dir, fine2mid_wnid, fine2coarse_wnid, fine_id2name, *args, **kw):
        super(HierarchyBREEDS2, self).__init__(*args, **kw)
        self.dataset_name = 'BREEDS'
        self.fine_names = list(set(self.targets))
        self.mid_names = sorted(list(set(fine2mid_wnid.values())))
        self.coarse_names = sorted(list(set(fine2coarse_wnid.values())))
        
        self.fine_id2name = fine_id2name # target id to name
        
        with open(os.path.join(info_dir, "node_names.txt")) as f:
            wnid2name = [l.strip().split('\t') for l in f.readlines()]
        
        name2wnid = dict() # name to wnid
        for pairs in wnid2name:
            wnid = pairs[0]
            name = pairs[1]
            if not name in name2wnid:
                name2wnid[name] = [wnid]
            else:
                name2wnid[name].append(wnid)
        self.name2wnid = name2wnid
        
        self.mid_map = [0] * len(self.fine_names)
        for fine_idx in range(len(set(self.class_to_idx.values()))):
            target_fine_name = self.fine_id2name[fine_idx]
            target_fine_wnids = self.name2wnid[target_fine_name]
            if len(target_fine_wnids) == 1:
                target_fine_wnid = target_fine_wnids[0]
            else:
                target_fine_wnid_candid = set(fine2mid_wnid.keys())
                target_fine_wnid = list(set(target_fine_wnids).intersection(target_fine_wnid_candid))[0]
            target_mid_wnid = fine2mid_wnid[target_fine_wnid]
            self.mid_map[fine_idx] = self.mid_names.index(target_mid_wnid)

        self.coarse_map = [0] * len(self.fine_names)
        for fine_idx in range(len(set(self.class_to_idx.values()))):
            target_fine_name = self.fine_id2name[fine_idx]
            target_fine_wnids = self.name2wnid[target_fine_name]
            if len(target_fine_wnids) == 1:
                target_fine_wnid = target_fine_wnids[0]
            else:
                target_fine_wnid_candid = set(fine2coarse_wnid.keys())
                target_fine_wnid = list(set(target_fine_wnids).intersection(target_fine_wnid_candid))[0]
            target_coarse_wnid = fine2coarse_wnid[target_fine_wnid]
            self.coarse_map[fine_idx] = self.coarse_names.index(target_coarse_wnid)
        self.mid_map = np.array(self.mid_map)
        self.coarse_map = np.array(self.coarse_map)
        self.coarsest_map = None
        self.mid2coarse = None
        self.mid2coarsest = None
    
    def __getitem__(self, index: int):
        # let get item be consistent with other datasets
        path, _ = self.samples[index]
        target_fine = int(self.targets[index])

        target_mid = int(self.mid_map[target_fine])
        target_coarse = int(self.coarse_map[target_fine])
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0, target_coarse, target_mid, target_fine

def make_dataloader(num_workers : int, batch_size : int, task : str, breeds_setting: str, 
                    info_dir : str = './dataset/breeds2/imagenet_class_hierarchy/modified', 
                    data_dir : str = '/data/common/ImageNet/ILSVRC12012') -> Tuple[DataLoader, DataLoader]:
    '''
    Args:
        breeds_setting : living17/entity13/entity30/nonliving26
    '''
    
    def make_levels_edge_mapping(info_dir, superclasses: list, levels: int):
        '''
            Backtrack 'levels' edges from each node to its ancestor.
        '''
        if levels < 1:
            raise ValueError("Levels must be a positive integer.")

        with open(os.path.join(info_dir, "class_hierarchy.txt")) as f:
            edges = [l.strip().split() for l in f.readlines()]
        parent_mapping = {end: start for start, end in edges}

        # Initialize mapping with the superclasses themselves
        ancestor_mapping = {sc: sc for sc in superclasses}

        # For each level, find the ancestor at that level
        for _ in range(levels):
            # Update each superclass's ancestor
            for sc in superclasses:
                # Proceed only if the ancestor is not the root (or has no parent)
                if ancestor_mapping[sc] in parent_mapping:
                    ancestor_mapping[sc] = parent_mapping[ancestor_mapping[sc]]

        return ancestor_mapping



    def make_loaders(info_dir, workers, batch_size, sup2_mapping, sup3_mapping, name_mapping, transforms, data_path, data_aug=True,
                custom_class=None, dataset="", label_mapping=None, subset=None,
                subset_type='rand', subset_start=0, val_batch_size=None,
                only_val=False, shuffle_train=True, shuffle_val=True, seed=1,
                custom_class_args=None):
        '''
            make PreHierarchyBREEDS2 train/test loaders.
        '''
        print(f"==> Preparing dataset {dataset}..")
        transform_train, transform_test = transforms
        if not data_aug:
            transform_train = transform_test

        if not val_batch_size:
            val_batch_size = batch_size

        if not custom_class:
            train_path = os.path.join(data_path, 'train')
            test_path = os.path.join(data_path, 'val')
            if not os.path.exists(test_path):
                test_path = os.path.join(data_path, 'test')

            if not os.path.exists(test_path):
                raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))
            
            if not only_val:
                train_set = HierarchyBREEDS2(info_dir, sup2_mapping, sup3_mapping, name_mapping, root=train_path, transform=transform_train,
                                            label_mapping=label_mapping)
            test_set = HierarchyBREEDS2(info_dir, sup2_mapping, sup3_mapping, name_mapping, root=test_path, transform=transform_test,
                                        label_mapping=label_mapping)
           
        else:
            if custom_class_args is None: custom_class_args = {}
            if not only_val:
                train_set = custom_class(root=data_path, train=True, download=True, 
                                    transform=transform_train, **custom_class_args)
            test_set = custom_class(root=data_path, train=False, download=True, 
                                    transform=transform_test, **custom_class_args)

        if not only_val:
            attrs = ["samples", "train_data", "data"]
            vals = {attr: hasattr(train_set, attr) for attr in attrs}
            assert any(vals.values()), f"dataset must expose one of {attrs}"
            train_sample_count = len(getattr(train_set,[k for k in vals if vals[k]][0]))

        if (not only_val) and (subset is not None) and (subset <= train_sample_count):
            assert not only_val
            if subset_type == 'rand':
                rng = np.random.RandomState(seed)
                subset = rng.choice(list(range(train_sample_count)), size=subset+subset_start, replace=False)
                subset = subset[subset_start:]
            elif subset_type == 'first':
                subset = np.arange(subset_start, subset_start + subset)
            else:
                subset = np.arange(train_sample_count - subset, train_sample_count)

            train_set = Subset(train_set, subset)

        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 0:
            train_sampler = DistributedSampler(dataset=train_set, num_replicas=int(os.environ['WORLD_SIZE']), rank = int(os.environ['RANK']), shuffle=True)
            train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, 
                                    num_workers=workers, pin_memory=True)
            
            test_sampler = DistributedSampler(dataset=test_set, num_replicas=int(os.environ['WORLD_SIZE']), rank = int(os.environ['RANK']), shuffle=False)
            test_loader = DataLoader(dataset=test_set, batch_size=val_batch_size, sampler=test_sampler, 
                                    num_workers=workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=workers, pin_memory=True)
            test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=shuffle_val, num_workers=workers, pin_memory=True)

        # if not only_val:
        #     train_loader = DataLoader(train_set, batch_size=batch_size, 
        #         shuffle=shuffle_train, num_workers=workers, pin_memory=True)

        # test_loader = DataLoader(test_set, batch_size=val_batch_size, 
        #         shuffle=shuffle_val, num_workers=workers, pin_memory=True)

        # if only_val:
        #     return None, test_loader


        return train_loader, test_loader

    if breeds_setting == 'living17':
        ret = make_living17(info_dir, split="rand")
        superclasses, subclass_split, label_map = ret
        sup2_mapping = make_levels_edge_mapping(info_dir, superclasses,1)
        sup3_mapping = make_levels_edge_mapping(info_dir, superclasses,2)
    elif breeds_setting == 'entity13':
        ret = make_entity13(info_dir, split="rand")
        superclasses, subclass_split, label_map = ret
        sup2_mapping = make_levels_edge_mapping(info_dir, superclasses,1)
        sup3_mapping = make_levels_edge_mapping(info_dir, superclasses,2)
    elif breeds_setting == 'entity30':
        ret = make_entity30(info_dir, split="rand")
        superclasses, subclass_split, label_map = ret
        sup2_mapping = make_levels_edge_mapping(info_dir, superclasses,2)
        sup3_mapping = make_levels_edge_mapping(info_dir, superclasses,3)
    elif breeds_setting == 'nonliving26':
        ret = make_nonliving26(info_dir, split="rand")
        superclasses, subclass_split, label_map = ret
        sup2_mapping = make_levels_edge_mapping(info_dir, superclasses,2)
        sup3_mapping = make_levels_edge_mapping(info_dir, superclasses,3)
    
    train_subclasses, test_subclasses = subclass_split

    dataset_source = CustomImageNet(data_dir, train_subclasses)
    
    source_transforms = (dataset_source.transform_train, dataset_source.transform_test)
    loaders_source = make_loaders(info_dir, num_workers, batch_size, sup2_mapping, sup3_mapping, label_map, transforms=source_transforms,
                                    data_path=dataset_source.data_path,
                                    dataset=dataset_source.ds_name,
                                    label_mapping=dataset_source.label_mapping,
                                    custom_class=dataset_source.custom_class,
                                    custom_class_args=dataset_source.custom_class_args)
    train_loader_source, val_loader_source = loaders_source # val_loader_source for s->s exp

    dataset_target = CustomImageNet(data_dir, test_subclasses)
    test_flattened_subclasses = [idx for split in test_subclasses for idx in split] # label index
    target_transforms = (dataset_target.transform_train, dataset_target.transform_test)
    loaders_target = make_loaders(info_dir, num_workers, batch_size, sup2_mapping, sup3_mapping, label_map, transforms=target_transforms,
                                    data_path=dataset_target.data_path,
                                    dataset=dataset_target.ds_name,
                                    label_mapping=dataset_target.label_mapping,
                                    custom_class=dataset_target.custom_class,
                                    custom_class_args=dataset_target.custom_class_args)
    train_loader_target, val_loader_target = loaders_target
    
    if task == 'ss':
        train_loader, test_loader = train_loader_source, val_loader_source
    elif task == 'tt': # where we use one shot setting for evaluation
        train_loader, test_loader = train_loader_target, val_loader_target
    elif task == 'st':
        train_loader, test_loader = train_loader_source, val_loader_target
    elif task == 'full': # use source/source as full data
        train_loader, test_loader = train_loader_source, val_loader_source
    return train_loader, test_loader