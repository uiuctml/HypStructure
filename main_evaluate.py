import sys
import os
import json
import pickle
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def load_coarse_map(dataset):
    if dataset == 'CIFAR10':
        return np.loadtxt('./cifar10/coarse_map.txt', dtype=int)
    elif dataset == 'CIFAR100':
        return np.loadtxt('./cifar100/coarse_map.txt', dtype=int)
    elif dataset == 'IMAGENET100':
        with open('./imagenet100/coarse_map.json', 'r') as f:
            coarse_map = json.load(f)
            return {int(k): v for k, v in coarse_map.items()}
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def main(dataset, model_save_location):
    if dataset == 'CIFAR10':
        ood_dataset = 'SVHN'
    elif dataset == 'CIFAR100':
        ood_dataset = 'SVHN'
    elif dataset == 'IMAGENET100':
        ood_dataset = 'iNaturalist'
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    coarse_map = load_coarse_map(dataset)
    features_path = os.path.join(model_save_location, ood_dataset)

    # Load data
    in_train_features = pickle.load(open(os.path.join(features_path, 'in_train_features_seed0.pkl'), 'rb'))
    in_train_fine_labels = pickle.load(open(os.path.join(features_path, 'in_train_labels_seed0.pkl'), 'rb'))
    in_train_coarse_labels = [coarse_map[i] for i in in_train_fine_labels]

    in_test_features = pickle.load(open(os.path.join(features_path, 'in_test_features_seed0.pkl'), 'rb'))
    in_test_fine_labels = pickle.load(open(os.path.join(features_path, 'in_test_labels_seed0.pkl'), 'rb'))
    in_test_coarse_labels = [coarse_map[i] for i in in_test_fine_labels]

    # Coarse accuracy
    cls = KNeighborsClassifier(50, metric="cosine").fit(in_train_features, in_train_coarse_labels)
    pred = cls.predict(in_test_features)
    acc = accuracy_score(in_test_coarse_labels, pred)
    print("Coarse Accuracy:", acc)

    # Fine accuracy
    cls = KNeighborsClassifier(50, metric="cosine").fit(in_train_features, in_train_fine_labels)
    pred = cls.predict(in_test_features)
    acc = accuracy_score(in_test_fine_labels, pred)
    print("Fine Accuracy:", acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evaluation of Fine and Coarse Classification accuracies.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to use (CIFAR10, CIFAR100, IMAGENET100)")
    parser.add_argument('--model_save_location', type=str, required=True, help="Path to the model save location")
    
    args = parser.parse_args()
    main(args.dataset, args.model_save_location)
