import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

from emd.emd_methods import emd, sinkhorn, swd, fft

class CPCCLoss(nn.Module):
    '''
    CPCC as a mini-batch regularizer.
    '''
    def __init__(self, dataset, metric, height=2, emd_weights=0, dist_weights=1):
        # make sure unique classes in layers[0] 
        super(CPCCLoss, self).__init__()
        self.metric = metric
        self.dataset = dataset
        self.fine2coarse = dataset.coarse_map
        if self.dataset.dataset_name == 'CIFAR100':
            self.fine2mid = dataset.mid_map
            self.fine2coarsest = dataset.coarsest_map
        self.height = height
        self.emd_weights = emd_weights
        self.dist_weights = dist_weights

    def forward(self, representations, target_fine):
        all_fine = torch.unique(target_fine)
        if len(all_fine) == 1: 
            return torch.tensor(1,device=target_fine.device)

        # class distance
        if self.metric == 'l2':
            target_fine_list = [torch.mean(torch.index_select(representations, 0, (target_fine == t).nonzero().flatten()),0) for t in all_fine]
            sorted_sums = torch.stack(target_fine_list, 0)
            pairwise_dist = F.pdist(sorted_sums, p=2.0)

        else:
            target_indices = [torch.where(target_fine == fine)[0] for fine in all_fine]
            combidx = [(target_indices[i], target_indices[j]) for (i, j) in combinations(range(len(all_fine)), 2)]

            def compute_inverse_weights(representations, target_indices):
                weights = []
                for indices in target_indices:
                    R = representations[indices]  # m * d matrix
                    centroid = R.mean(dim=0)  # d-dimensional centroid
                    dists = torch.norm(R - centroid, dim=1)  # Euclidean distances (m-dimensional vector)
                    weight = F.softmax(-dists, dim=0)  # Normalize distances using softmax into probability distribution
                    weights.append(weight)
                return weights
        
            def compute_weights(representations, target_indices):
                weights = []
                for indices in target_indices:
                    R = representations[indices] 
                    centroid = R.mean(dim=0)  
                    dists = torch.norm(R - centroid, dim=1)  
                    weight = F.softmax(dists, dim=0)  # positive dist
                    weights.append(weight)
                return weights

            if self.emd_weights == 1:
                weightX_list = compute_weights(representations, [pair[0] for pair in combidx])
                weightY_list = compute_weights(representations, [pair[1] for pair in combidx])
            elif self.emd_weights == 2:
                weightX_list = compute_inverse_weights(representations, [pair[0] for pair in combidx])
                weightY_list = compute_inverse_weights(representations, [pair[1] for pair in combidx])
            else:
                weightX_list = [[] for _ in combidx]
                weightY_list = [[] for _ in combidx]

            if self.metric == 'emd':  # emd
                all_pairwise = torch.cdist(representations, representations)
                dist_matrices = [all_pairwise.index_select(0, pair[0]).index_select(1, pair[1]) for pair in combidx]
                pairwise_dist = torch.stack([emd(M, weightX_list[i], weightY_list[i]) for i, M in enumerate(dist_matrices)])
            elif self.metric == 'sk':  # sinkhorn
                all_pairwise = torch.cdist(representations, representations)
                dist_matrices = [all_pairwise.index_select(0, pair[0]).index_select(1, pair[1]) for pair in combidx]
                pairwise_dist = torch.stack([sinkhorn(M) for M in dist_matrices])
            elif self.metric == 'swd':  # swd
                pairwise_dist = torch.stack([swd(representations[pair[0]], representations[pair[1]]) for pair in combidx])
            elif self.metric == 'fft':  # FastFT
                pairwise_dist = torch.stack([fft(representations[pair[0]], representations[pair[1]], weightX_list[i], weightY_list[i]) for i, pair in enumerate(combidx)])
            else:
                raise NotImplementedError
        
        if self.height == 2:
            tree_pairwise_dist = self.two_level_dT(all_fine, self.fine2coarse, pairwise_dist.device)
        elif self.height == 3:
            tree_pairwise_dist = self.three_level_dT(all_fine, self.fine2mid, self.fine2coarse, pairwise_dist.device)
        elif self.height == 4:
            tree_pairwise_dist = self.four_level_dT(all_fine, self.fine2mid, self.fine2coarse, self.fine2coarsest, pairwise_dist.device)
        
        res = 1 - torch.corrcoef(torch.stack([pairwise_dist, tree_pairwise_dist], 0))[0,1] # maximize cpcc
       
        if torch.isnan(res): 
            return torch.tensor(1,device=pairwise_dist.device)
        else:
            return res

    def two_level_dT(self, all_fine, fine2layer, device):
        if self.dataset.dataset_name == 'INAT':
            tree_pairwise_dist = torch.tensor([2 if fine2layer[all_fine[i]] == fine2layer[all_fine[j]] 
                                            else 6 for (i,j) in combinations(range(len(all_fine)),2)], 
                                            device=device)
        elif self.dataset.dataset_name == 'CIFAR100':
            tree_pairwise_dist = []
            for (i,j) in combinations(range(len(all_fine)),2):
                if fine2layer[all_fine[i]] == fine2layer[all_fine[j]]:
                    if fine2layer[all_fine[i]] == 0:
                        tree_pairwise_dist.append(2*self.dist_weights)
                    else:
                        tree_pairwise_dist.append(2)
                else:
                    tree_pairwise_dist.append(4)
            tree_pairwise_dist = torch.as_tensor(tree_pairwise_dist, device=device) 
        else:
            tree_pairwise_dist = torch.tensor([2 if fine2layer[all_fine[i]] == fine2layer[all_fine[j]] 
                                            else 4 for (i,j) in combinations(range(len(all_fine)),2)], 
                                            device=device)
        return tree_pairwise_dist
    
    
    def three_level_dT(self, all_fine, fine2mid, fine2coarse, device):
        tree_pairwise_dist = []
        for (i,j) in combinations(range(len(all_fine)),2):
            fine_label_i = all_fine[i]
            fine_label_j = all_fine[j]
            if fine2mid[fine_label_i] == fine2mid[fine_label_j]:
                tree_pairwise_dist.append(2)
            elif fine2coarse[fine_label_i] == fine2coarse[fine_label_j]:
                tree_pairwise_dist.append(4)
            else:
                tree_pairwise_dist.append(6)
        return torch.as_tensor(tree_pairwise_dist, device=device)

    def four_level_dT(self, all_fine, fine2mid, fine2coarse, fine2coarsest, device):
        tree_pairwise_dist = []
        for (i,j) in combinations(range(len(all_fine)),2):
            fine_label_i = all_fine[i]
            fine_label_j = all_fine[j]
            if fine2mid[fine_label_i] == fine2mid[fine_label_j]:
                tree_pairwise_dist.append(2)
            elif fine2coarse[fine_label_i] == fine2coarse[fine_label_j]:
                tree_pairwise_dist.append(4)
            elif fine2coarsest[fine_label_i] == fine2coarsest[fine_label_j]:
                tree_pairwise_dist.append(6)
            else:
                tree_pairwise_dist.append(8)
        return torch.as_tensor(tree_pairwise_dist, device=device)