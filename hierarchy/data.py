from typing import *
import numpy as np

import networkx as nx
import os
import json

class Hierarchy():
    '''
    Class representing a general ImageNet-style hierarchy, modified from 
    https://github.com/MadryLab/robustness/blob/master/robustness/tools/breeds_helpers.py 
    '''
    def __init__(self, dataset_name, info_dir='/path/to/repository/', *args, **kw):
        """
        Args:
            info_dir (str) : Base path to datasets and hierarchy information files. Contains a 
                "class_hierarchy.txt" file with one edge per line, a
                "node_names.txt" mapping nodes to names, and "dataset_class_info.json".
        """
        super(Hierarchy, self).__init__(*args, **kw)
        
        self.dataset_name = dataset_name
        self.info_dir = info_dir
        
        REQUIRED_FILES = [f'{self.dataset_name}/dataset_class_info.json',
                  f'{self.dataset_name}/class_hierarchy.txt',
                  f'{self.dataset_name}/node_names.txt']

        for f in REQUIRED_FILES:
            if not os.path.exists(os.path.join(self.info_dir, f)):
                self.generate_files()
            
        # Details about dataset class names (leaves), IDS
        with open(os.path.join(self.info_dir, f"{self.dataset_name}/dataset_class_info.json")) as f:
            class_info = json.load(f)

        # Hierarchy represented as edges between parent & child nodes.
        with open(os.path.join(self.info_dir, f'{self.dataset_name}/class_hierarchy.txt')) as f:
            edges = [l.strip().split() for l in f.readlines()]

        # Information (names, IDs) for intermediate nodes in hierarchy.
        with open(os.path.join(self.info_dir,  f'{self.dataset_name}/node_names.txt')) as f:
            mapping = [l.strip().split('\t') for l in f.readlines()]


        # Original dataset classes
        self.LEAF_IDS = [c[1] for c in class_info] # wnid
        self.LEAF_ID_TO_NAME = {c[1]: c[2] for c in class_info} # wnid : name
        self.LEAF_ID_TO_NUM = {c[1]: c[0] for c in class_info} # wnid : labelid
        self.LEAF_NUM_TO_NAME = {c[0]: c[2] for c in class_info} # labelid : name

        # Full hierarchy
        self.HIER_NODE_NAME = {w[0]: w[1] for w in mapping} # wnid : name
        self.NAME_TO_NODE_ID = {w[1]: w[0] for w in mapping} # name : wnid
        self.graph = self._make_parent_graph(self.LEAF_IDS, edges)

        # make label mapping
        self.label_map = self.get_label_mapping()
        # generate tree distance
        self.tree_dist = self.generate_tree_dist()
        # leave node names
        self.leaf_names = [c[2] for c in class_info] 
        


    @staticmethod
    def _make_parent_graph(nodes, edges):
        """
        Obtain networkx representation of class hierarchy.

        Args:
            nodes [str] : List of node names to traverse upwards.
            edges [(str, str)] : Tuples of parent-child pairs.

        Return:
            networkx representation of the graph.
        """

        # create full graph
        full_graph_dir = {}
        for p, c in edges:
            if p not in full_graph_dir:
                full_graph_dir[p] = {c: 1}
            else:
                full_graph_dir[p].update({c: 1})
                    
        FG = nx.DiGraph(full_graph_dir)

        # perform backward BFS to get the relevant graph
        graph_dir = {}
        todo = [n for n in nodes if n in FG.nodes()] # skip nodes not in graph
        while todo:
            curr = todo
            todo = []
            for w in curr:
                for p in FG.predecessors(w):
                    if p not in graph_dir:
                        graph_dir[p] = {w: 1}
                    else:
                        graph_dir[p].update({w: 1})
                    todo.append(p)
            todo = set(todo)
        
        return nx.DiGraph(graph_dir)

    def get_root_node(self):
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                return node
    
    def get_ancestors(self, node): # leaf to root path
        return nx.shortest_path(self.graph, source=self.get_root_node(), target=node)[::-1]
    
    def find_leaf_nodes(self):
        return [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
    
    def get_label_mapping(self):
        leaf_nodes = self.find_leaf_nodes()
        leaf_ancestors = {}
        max_path = -1
        for leaf in leaf_nodes:
            ancestors = self.get_ancestors(leaf)
            # here leaf is wnid
            # convert to label id
            if len(ancestors) > max_path:
                max_path = len(ancestors)
            leaf_ancestors[self.LEAF_ID_TO_NUM[leaf]] = ancestors
        
        label_map = np.empty((len(leaf_nodes), max_path),dtype=object)
        for leaf in leaf_nodes:
            true_path = leaf_ancestors[self.LEAF_ID_TO_NUM[leaf]]
            padded_path = [''] * (max_path - len(true_path)) + true_path
            label_map[self.LEAF_ID_TO_NUM[leaf]] = padded_path

        return label_map


    def generate_tree_dist(self):
        distances = {}
        all_distances = dict(nx.all_pairs_shortest_path_length(self.graph.to_undirected()))
        for node1, paths in all_distances.items():
            for node2, dist in paths.items():
                if node1 != node2:
                    node_pair = (node1, node2)
                    rev_pair = (node2, node1)
                    if node_pair not in distances:
                        distances[node_pair] = dist
                        distances[rev_pair] = dist
        return distances
    
    
    def generate_files(self):
        # specify how to generate files for each dataset
        return None

