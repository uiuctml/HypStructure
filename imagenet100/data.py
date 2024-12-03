from typing import *
from torchvision.datasets import ImageFolder
from hierarchy.data import Hierarchy

import json, os

class ImageNet100(ImageFolder):
    def __init__(self, root, train = True, *args, **kw):
        if train: 
            super(ImageNet100, self).__init__(root = os.path.join(root, 'train'), *args, **kw)
        else:
            super(ImageNet100, self).__init__(root = os.path.join(root, 'val'), *args, **kw)

class HierarchyImageNet100(Hierarchy, ImageNet100):
    def __init__(self, *args, **kw):
        super(HierarchyImageNet100, self).__init__(dataset_name='imagenet100', *args, **kw)

    def generate_files(self):
        # Use class_hierarchy.txt, node_names.txt for original wordnet hierarchy in BREEDS.
        # For 100 leaf nodes, use https://github.com/deeplearning-wisc/MCM/blob/main/data/ImageNet100/class_list.txt
        with open(os.path.join(self.info_dir,  f'{self.dataset_name}/class_list.txt')) as f:
           leaf_names =  [l.strip() for l in f.readlines()]

        with open(os.path.join(self.info_dir,  f'{self.dataset_name}/node_names.txt')) as f:
            mapping = [l.strip().split('\t') for l in f.readlines()]
        HIER_NODE_NAME = {w[0]: w[1] for w in mapping} # wnid : name

        # dataset_class_info.json
        # a list, each entry is [int(label_id), wordnet id, label name]
        # all leaves
        leaf2labelid = {class_name : self.class_to_idx[class_name] for class_name in self.classes}
        data_class_info = [[leaf2labelid[leaf],leaf,HIER_NODE_NAME[leaf]] for leaf in leaf_names]
        with open(os.path.join(self.info_dir, f"{self.dataset_name}/dataset_class_info.json"), 'w') as file:
            json.dump(data_class_info, file)


