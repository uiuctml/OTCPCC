from typing import *

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
# from torchvision.datasets import INaturalist
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

# from hierarchy.data import Hierarchy, get_k_shot

import os
import json

import os.path
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


ImageFile.LOAD_TRUNCATED_IMAGES = True

CATEGORIES_2021 = ["kingdom", "phylum", "class", "order", "family", "genus"]

class INaturalist(VisionDataset):
    # https://pytorch.org/vision/main/_modules/torchvision/datasets/inaturalist.html#INaturalist
    # optimize RAM

    def __init__(
        self,
        root: Union[str, Path],
        version: str = "2021_train",
        target_type: Union[List[str], str] = "full",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(os.path.join(root, version), transform=transform, target_transform=target_transform)
        self.dataset_name = 'INAT'
        self.all_categories: List[str] = []

        # map: category type -> name of category -> index
        self.categories_index: Dict[str, Dict[str, int]] = {}

        # list indexed by category id, containing mapping from category type -> index
        self.categories_map: List[Dict[str, int]] = []

        self.target_type = target_type
        self._init_2021()

        # index of all files: (full category id, filename)
        self.index: List[Tuple[int, str]] = []

        for dir_index, dir_name in enumerate(self.all_categories):
            files = os.listdir(os.path.join(self.root, dir_name))
            for fname in files:
                self.index.append((dir_index, fname))

        self._cast_data_structures()

    def _init_2021(self) -> None:
        """Initialize based on 2021 layout"""

        self.all_categories = sorted(os.listdir(self.root))

        # map: category type -> name of category -> index
        self.categories_index = {k: {} for k in CATEGORIES_2021}

        for dir_index, dir_name in enumerate(self.all_categories):
            pieces = dir_name.split("_")
            if len(pieces) != 8:
                raise RuntimeError(f"Unexpected category name {dir_name}, wrong number of pieces")
            if pieces[0] != f"{dir_index:05d}":
                raise RuntimeError(f"Unexpected category id {pieces[0]}, expecting {dir_index:05d}")
            cat_map = {}
            for cat, name in zip(CATEGORIES_2021, pieces[1:7]):
                if name in self.categories_index[cat]:
                    cat_id = self.categories_index[cat][name]
                else:
                    cat_id = len(self.categories_index[cat])
                    self.categories_index[cat][name] = cat_id
                cat_map[cat] = cat_id
            self.categories_map.append(cat_map)

    def _cast_data_structures(self) -> None:
        """Cast categories_index to DataFrame of DataFrames and categories_map to NumPy array of DataFrames"""
        self.all_categories = np.array(self.all_categories, dtype=object)

        # Cast self.categories_index
        # for key in self.categories_index.keys():
        #     self.categories_index[key] = pd.DataFrame(list(self.categories_index[key].items()), columns=["name", "index"])
        self.categories_index = None # we don't need this anymore

        # Cast self.categories_map
        self.categories_map_names = np.array(list(self.categories_map[0].keys())) # kingdom/phylum/class/order/family/genus
        self.categories_map = np.array([[cat_map[key] for key in self.categories_map_names] for cat_map in self.categories_map])

        # Cast self.index
        self.index = np.array(self.index, dtype=object)

    def __len__(self) -> int:
        return len(self.index)



class HierarchyINaturalist(INaturalist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)             
        # self.index contains species label
        self.targets = np.array([self.index[idx][0] for idx in range(len(self.index))])
        
        idx_fine = np.where(self.categories_map_names == 'class')[0][0]
        species2fine = self.categories_map[:, idx_fine] # species -> fine
        idx_coarse = np.where(self.categories_map_names == 'phylum')[0][0]
        species2coarse = self.categories_map[:, idx_coarse] # species target -> coarse target
        # convert self.index to fine level label
        self.targets = species2fine[self.targets] # classify class
        

        self.fine_map = np.arange(len(np.unique(self.targets)))
        self.fine_names = self.fine_map.astype(str)
        
        unique_fine, unique_indices = np.unique(species2fine, return_index=True)
        coarse_values = species2coarse[unique_indices]
        sorted_indices = np.argsort(unique_fine)
        self.coarse_map = coarse_values[sorted_indices]
        self.coarse_names = np.unique(self.coarse_map).astype(str)
        
        self.img_size = 224
        self.channel = 3

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        cat_id, fname = self.index[idx]
        img = Image.open(os.path.join(self.root, self.all_categories[cat_id], fname))

        if self.transform is not None:
            img = self.transform(img)

        target_fine = self.targets[idx]
        target_mid = -1
        target_coarse = self.coarse_map[target_fine]
        target_coarsest = -1

        return img, target_coarsest, target_coarse, target_mid, target_fine


class HierarchyINaturalistSubset(HierarchyINaturalist):
    def __init__(self, indices : List[int], fine_classes : List[int], *args, **kw):
        super(HierarchyINaturalistSubset, self).__init__(*args, **kw)
        self.index = self.index[indices]
        old_targets = self.targets[indices] # old fine targets, sliced. index -> old fine
        fine_classes = np.sort(fine_classes) # fine class id in HierarchyINAT index
        self.fine_names = self.fine_names[fine_classes] # old fine names, sliced
        self.fine_map = np.arange(len(fine_classes)) # number of fine classes subset. new fine
        self.targets = np.searchsorted(fine_classes, old_targets) # reset fine target id from 0. index -> new fine


        # reset other hierarchy level index, from 0
        old_coarse_map = self.coarse_map[fine_classes] # subset of coarse fine map. fine->coarse
        coarse_classes = np.unique(old_coarse_map)
        self.coarse_names = self.coarse_names[coarse_classes]
        self.coarse_map = np.searchsorted(coarse_classes, old_coarse_map) # argsort

        old_mid_map = None
        mid_classes = None
        self.mid_names = None
        self.mid_map = None

        old_coarsest_map = None
        coarsest_classes = None
        self.coarsest_names = None
        self.coarsest_map = None

    def __len__(self):
        return len(self.index)
    

def make_dataloader(num_workers : int, batch_size : int, task : str) -> Tuple[DataLoader, DataLoader]:
    def make_full_dataset() -> Tuple[Dataset, Dataset]:
        train_dataset = HierarchyINaturalist(root = '/data/common/iNaturalist/inat-mini-val',
                                                version = '2021_train_mini', 
                                                target_type = ['full'],                     
                                                transform = transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        # not augment test set except of normalization
        test_dataset = HierarchyINaturalist(root = '/data/common/iNaturalist/inat-mini-val',
                                                version = '2021_valid', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        return train_dataset, test_dataset

    def make_subpopsplit_dataset(train_dataset, test_dataset, task : str) -> Tuple[DataLoader, DataLoader]: 
        subset_info_path = "./dataset/inat/subset_info.json"

        if os.path.exists(subset_info_path):
            # Load subset information
            with open(subset_info_path, "r") as f:
                subset_info = json.load(f)
            
            idx_train_source = subset_info["idx_train_source"]
            idx_train_target = subset_info["idx_train_target"]
            source_fine_cls = subset_info["source_fine_cls"]
            target_fine_cls = subset_info["target_fine_cls"]
            idx_test_source = subset_info["idx_test_source"]
            idx_test_target = subset_info["idx_test_target"]

        else:
            train_all_fine_map = train_dataset.targets
            train_sortidx = np.argsort(train_all_fine_map)
            train_sorted_fine_map = np.array(train_all_fine_map)[train_sortidx]
            # a dictionary that maps coarse id to a list of fine id
            target_fine_dict = {i:[] for i in range(len(train_dataset.coarse_names))}
            idx_train_source = [] # index of image (based on original Pytorch CIFAR dataset) that sends to source
            idx_train_target = []
            f2c = dict(zip(range(len(train_dataset.fine_names)),train_dataset.coarse_map))
            
            # finish c2f, know each coarse -> which set of fine class
            # keep 60% in source, 40% in target
            c2f = {}

            for fine_id, coarse_id in enumerate(train_dataset.coarse_map):
                if coarse_id not in c2f:
                    c2f[coarse_id] = []
                c2f[coarse_id].append(fine_id)
            coarse_counts = {coarse_id: len(fine_ids) for coarse_id, fine_ids in c2f.items()}

            # remove coarse and fine id where number of fine classes = 1, 2,
            # since 40-60 split will cause empty source/train coarse id
            small_coarse_set = {key for key, count in coarse_counts.items() if count <= 2}
            
            for idx in tqdm(range(len(train_sortidx)), desc="splitting indices"): # loop thru all argsort fine
                coarse_id = f2c[train_sorted_fine_map[idx]]
                if coarse_id not in small_coarse_set:
                    target_fine_dict[coarse_id].append(train_sorted_fine_map[idx])
                    if len(set(target_fine_dict[coarse_id])) <= int(0.4 * coarse_counts[coarse_id]): 
                        # 40% to few shot second stage
                        idx_train_target.append(train_sortidx[idx])
                    else:
                        # For the rest of images,
                        # send to source
                        idx_train_source.append(train_sortidx[idx])
            
            for key in target_fine_dict:
                target = target_fine_dict[key] # fine label id for [coarse]
                d = {x: True for x in target}
                target_fine_dict[key] = list(d.keys())[:int(0.4 * coarse_counts[key])] # all UNIQUE fine classes sent to target for [coarse]

            target_fine_cls = [] # all 40% fine classes sent to target
            for key in target_fine_dict:
                target_fine_cls.extend(target_fine_dict[key])

            test_all_fine_map = test_dataset.targets
            idx_test_source = []
            idx_test_target = []
            for idx in range(len(test_all_fine_map)):
                fine_id = test_all_fine_map[idx]
                coarse_id = f2c[fine_id]
                if fine_id in target_fine_dict[coarse_id]:
                    idx_test_target.append(idx)
                else:
                    idx_test_source.append(idx)

            source_fine_cls = list(set(range(len(train_dataset.fine_names))) - set(target_fine_cls))

            subset_info = {
                'idx_train_source': [int(i) for i in idx_train_source],
                'idx_train_target': [int(i) for i in idx_train_target],
                'source_fine_cls': [int(i) for i in source_fine_cls],
                'target_fine_cls': [int(i) for i in target_fine_cls],
                'idx_test_source': [int(i) for i in idx_test_source],
                'idx_test_target': [int(i) for i in idx_test_target]
            }
            with open(subset_info_path, "w") as f:
                json.dump(subset_info, f)
        source_train = HierarchyINaturalistSubset(idx_train_source, source_fine_cls, 
                                                  root = '/data/common/iNaturalist/inat-mini-val',
                                                  version = '2021_train_mini', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))                              
        source_test = HierarchyINaturalistSubset(idx_test_source, source_fine_cls, 
                                                  root = '/data/common/iNaturalist/inat-mini-val',
                                                  version = '2021_valid', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        target_train = HierarchyINaturalistSubset(idx_train_target, target_fine_cls, 
                                                  root = '/data/common/iNaturalist/inat-mini-val',
                                                  version = '2021_train_mini', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        target_test = HierarchyINaturalistSubset(idx_test_target, target_fine_cls, 
                                                  root = '/data/common/iNaturalist/inat-mini-val',
                                                  version = '2021_valid', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        if task == 'ss':
            train_dataset, test_dataset = source_train, source_test
        elif task == 'st':
            train_dataset, test_dataset = source_train, target_test
        elif task == 'ts':
            train_dataset, test_dataset = target_train, source_test
        elif task == 'tt':
            train_dataset, test_dataset = target_train, target_test
        return train_dataset, test_dataset

    init_train_dataset, init_test_dataset = make_full_dataset()
    train_dataset, test_dataset = make_subpopsplit_dataset(init_train_dataset, init_test_dataset, task)

    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 0:
        train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=int(os.environ['WORLD_SIZE']), rank = int(os.environ['RANK']), shuffle=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, 
                                 num_workers=num_workers, pin_memory=True)
        
        test_sampler = DistributedSampler(dataset=test_dataset, num_replicas=int(os.environ['WORLD_SIZE']), rank = int(os.environ['RANK']), shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=test_sampler, 
                                num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

