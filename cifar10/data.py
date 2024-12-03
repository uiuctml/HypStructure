from typing import *
from PIL import Image
from torchvision.datasets import CIFAR10
from hierarchy.data import Hierarchy

import json, os



class HierarchyCIFAR10(Hierarchy, CIFAR10):
    def __init__(self, *args, **kw):
        super(HierarchyCIFAR10, self).__init__(dataset_name='cifar10', *args, **kw)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def generate_files(self):
        
        hierarchy = {"root" : ['transportation','animal'],
                     "transportation" : ['airplane','automobile','ship','truck'],
                     "animal" : ['bird','cat','deer','dog','frog','horse'],
                    }
       

        keys = set(hierarchy.keys())
        leaf_names = set()

        for value_list in hierarchy.values():
            for value in value_list:
                if value not in keys:
                    leaf_names.add(value)

        leaf_names = sorted(list(leaf_names))


        all_names = sorted(list(hierarchy.keys()) + leaf_names)
        name2wnid = {all_names[i]: str(i) for i in range(len(all_names))}
        leaf2labelid = {leaf_names[i]: i for i in range(len(leaf_names))}

        edges = []
        for parent, children in hierarchy.items():
            parent_wnid = name2wnid[parent]
            for child in children:
                child_wnid = name2wnid[child]
                edges.append((parent_wnid, child_wnid))

        with open(os.path.join(self.info_dir, f'{self.dataset_name}/class_hierarchy.txt'), 'w') as file:
            for id1, id2 in edges:
                file.write(f"{id1} {id2}\n")

        data_class_info = [[leaf2labelid[leaf],name2wnid[leaf],leaf] for leaf in leaf_names]
        with open(os.path.join(self.info_dir, f"{self.dataset_name}/dataset_class_info.json"), 'w') as file:
            json.dump(data_class_info, file)

        with open(os.path.join(self.info_dir,  f'{self.dataset_name}/node_names.txt'), 'w') as file:
            for key, value in name2wnid.items():
                file.write(f"{value}\t{key}\n")
