import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from itertools import combinations
from typing import *

from geom import poincare, hyperboloid
import geom.pmath as pmath
import geom.nn as hypnn
from geom.pmath import dist_matrix

class CPCCLoss(nn.Module):
    '''
    CPCC as a mini-batch regularizer.
    '''
    def __init__(self, dataset, args):
        super(CPCCLoss, self).__init__()
        self.tree_dist = dataset.tree_dist
        self.label_map_str = dataset.label_map # fine label id => wnid from leaf to root
        if args.leaf_only:
            self.label_map_str = dataset.label_map[:,0].reshape(-1,1)
        self.label_map_int, self.str2int, self.int2str = self.map_strings_to_integers(self.label_map_str)
        self.empty_int = self.str2int.get('',-1) # integer id for empty string used for padding in label map
        self.tree_depth = len(self.label_map_str[0])
        self.distance_type = args.cpcc_metric
        if self.distance_type == 'poincare_exp_c':
            self.to_hyperbolic = hypnn.ToPoincare(c=args.hyp_c, ball_dim=args.poincare_head_dim, riemannian=True, clip_r=args.clip_r, train_c=args.train_c)
            self.dist_f = lambda x, y: dist_matrix(x, y, c=args.hyp_c)
    
    def map_strings_to_integers(self, string_array):
        string_to_int = {}
        int_to_string = {}
        current_id = 0 # nonnegative

        int_array = np.zeros(string_array.shape, dtype=int)

        for i in range(string_array.shape[0]):
            for j in range(string_array.shape[1]):
                string = string_array[i, j]
                if string not in string_to_int:
                    string_to_int[string] = current_id
                    int_to_string[current_id] = string
                    current_id += 1
                int_array[i, j] = string_to_int[string]

        return int_array, string_to_int, int_to_string
        
    def forward(self, representations, targets_fine):
        label_map = torch.tensor(self.label_map_int,device=targets_fine.device)
        # assume the tree has d levels
        # targets: B * d, d = 0 => leaf node, d = d-1 => root
        targets = label_map[targets_fine]
        all_unique_int = [torch.unique(targets[:, col][targets[:, col] != self.empty_int]) for col in range(targets.shape[1])] # unique node for each level
        all_unique_str = [] # flattened all unique string nodes in this batch
        target_mean_list = [] # d components from fine to coarse
        if self.distance_type == 'poincare_mean':
            representations_poincare = poincare.expmap0(representations)
        for col, unique_values in enumerate(all_unique_int):
            for val in unique_values:
                if self.distance_type == 'poincare_mean':
                    column_mean = pmath.poincare_mean(torch.index_select(representations_poincare, 0, (targets[:, col] == val).nonzero(as_tuple=True)[0]), dim=0, c=1.0)
                else:
                    column_mean = torch.mean(torch.index_select(representations, 0, (targets[:, col] == val).nonzero(as_tuple=True)[0]), 0)
                target_mean_list.append(column_mean)
                all_unique_str.append(self.int2str[val.item()])
        sorted_sums = torch.stack(target_mean_list, 0)

        if self.distance_type == 'l2':
            pairwise_dist = F.pdist(sorted_sums, p=2.0) # get pairwise distance
        elif self.distance_type == 'nl2':
            # normalized
            all_norms = torch.norm(sorted_sums, dim=1, p=2).unsqueeze(-1)
            pairwise_dist = F.pdist(all_norms, p=2.0)
        elif self.distance_type == 'l1':
            pairwise_dist = F.pdist(sorted_sums, p=1.0)
        elif self.distance_type == 'poincare':
            # Project into the poincare ball with norm <= 1 - epsilon
            # https://www.tensorflow.org/addons/api_docs/python/tfa/layers/PoincareNormalize
            epsilon = 1e-5 
            all_norms = torch.norm(sorted_sums, dim=1, p=2).unsqueeze(-1)
            normalized_sorted_sums = sorted_sums * (1 - epsilon) / all_norms
            all_normalized_norms = torch.norm(normalized_sorted_sums, dim=1, p=2) 
            # |u-v|^2
            condensed_idx = torch.triu_indices(len(all_unique_str), len(all_unique_str), offset=1, device = sorted_sums.device)
            numerator_square = torch.sum((normalized_sorted_sums[None, :] - normalized_sorted_sums[:, None])**2, -1)
            numerator = numerator_square[condensed_idx[0],condensed_idx[1]]
            # (1 - |u|^2) * (1 - |v|^2)
            denominator_square = ((1 - all_normalized_norms**2).reshape(-1,1)) @ (1 - all_normalized_norms**2).reshape(1,-1)
            denominator = denominator_square[condensed_idx[0],condensed_idx[1]]
            pairwise_dist = torch.acosh(1 + 2 * (numerator/denominator))
        elif self.distance_type == 'poincare_exp':
            sorted_sums_exp = poincare.expmap0(sorted_sums)
            condensed_idx = torch.triu_indices(len(all_unique_str), len(all_unique_str), offset=1, device = sorted_sums.device)
            pairwise_dists_poincare_matrix = poincare.pairwise_distance(sorted_sums_exp)
            pairwise_dist = pairwise_dists_poincare_matrix[condensed_idx[0],condensed_idx[1]]
        elif self.distance_type == 'poincare_exp_c':
            sorted_sums_exp = self.to_hyperbolic(sorted_sums)
            condensed_idx = torch.triu_indices(len(all_unique_str), len(all_unique_str), offset=1, device = sorted_sums.device)
            pairwise_dists_poincare_matrix = self.dist_f(sorted_sums_exp, sorted_sums_exp)
            pairwise_dist = pairwise_dists_poincare_matrix[condensed_idx[0],condensed_idx[1]]
        elif self.distance_type == 'poincare_mean':
            condensed_idx = torch.triu_indices(len(all_unique_str), len(all_unique_str), offset=1, device = sorted_sums.device)
            pairwise_dists_poincare_matrix = poincare.pairwise_distance(sorted_sums)
            pairwise_dist = pairwise_dists_poincare_matrix[condensed_idx[0],condensed_idx[1]]

        tree_pairwise_dist = self.dT(all_unique_str, pairwise_dist.device)
        
        res = 1 - torch.corrcoef(torch.stack([pairwise_dist, tree_pairwise_dist], 0))[0,1] # maximize cpcc
        if torch.isnan(res):
            return torch.tensor(1,device=pairwise_dist.device)
        else:
            return res

    def dT(self, all_node, device): 
        tree_pairwise_dist = []
        for i in range(len(all_node)):
            for j in range(i+1, len(all_node)):
                tree_pairwise_dist.append(self.tree_dist[(all_node[i], all_node[j])])
        return torch.tensor(tree_pairwise_dist, device=device)

"""
Adapted from SupCon: https://github.com/HobbitLong/SupContrast/
"""

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss