from typing import *
from PIL import Image
from torchvision.datasets import CIFAR100
from hierarchy.data import Hierarchy

import json, os



class HierarchyCIFAR100(Hierarchy, CIFAR100):
    def __init__(self, *args, **kw):
        super(HierarchyCIFAR100, self).__init__(dataset_name='cifar100', *args, **kw)
       
    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def generate_files(self):
        # Following the convention from : https://github.com/MadryLab/BREEDS-Benchmarks/tree/master/imagenet_class_hierarchy/modified

        # Step 1: 
        # wordnet id = unique id for all nodes
        # assume each node has a unique name
        # hierarchy: parent -> children
        hierarchy = {"root" : ["aquatic_mammals", "fish", "flowers", "food_containers","fruit_and_vegetables",
                               "household_electrical_devices", "household_furniture", "insects", "large_carnivores", "large_man-made_outdoor_things",
                               "large_natural_outdoor_scenes","large_omnivores_and_herbivores","medium_mammals","non-insect_invertebrates","people",
                               "reptiles","small_mammals","trees","vehicles_1","vehicles_2"],
                    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
                    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                    "flowers": ["orchids", "poppies", "roses", "sunflowers", "tulips"],
                    "food_containers": ["bottles", "bowls", "cans", "cups", "plates"],
                    "fruit_and_vegetables": ["apples", "mushrooms", "oranges", "pears", "sweet_peppers"],
                    "household_electrical_devices": ["clock", "computer_keyboard", "lamp", "telephone", "television"],
                    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
                    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
                    "large_man-made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
                    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
                    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
                    "non-insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
                    "people": ["baby", "boy", "girl", "man", "woman"],
                    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
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

        
        # Step 2:
        # label_id = id from the target attribute of the orginal torchvision dataset
        leaf2labelid = {leaf_names[i]: i for i in range(len(leaf_names))}

        # Step 3:
        # label name = leaf_names
        # parent wordnet id \space children wordnet id
        
        edges = []
        for parent, children in hierarchy.items():
            parent_wnid = name2wnid[parent]
            for child in children:
                child_wnid = name2wnid[child]
                edges.append((parent_wnid, child_wnid))

        with open(os.path.join(self.info_dir, f'{self.dataset_name}/class_hierarchy.txt'), 'w') as file:
            for id1, id2 in edges:
                file.write(f"{id1} {id2}\n")


        # dataset_class_info.json
        # a list, each entry is [int(label_id), wordnet id, label name]
        # all leaves
        data_class_info = [[leaf2labelid[leaf],name2wnid[leaf],leaf] for leaf in leaf_names]
        with open(os.path.join(self.info_dir, f"{self.dataset_name}/dataset_class_info.json"), 'w') as file:
            json.dump(data_class_info, file)

        # node_names.txt
        # wordnet id \tab node name
        with open(os.path.join(self.info_dir,  f'{self.dataset_name}/node_names.txt'), 'w') as file:
            for key, value in name2wnid.items():
                file.write(f"{value}\t{key}\n")
