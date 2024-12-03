import os
import random

import torch
import numpy as np
import faiss
import sklearn.metrics as skm
from sklearn.covariance import ledoit_wolf
from sklearn.neighbors import KNeighborsClassifier

from geom.frechet import Frechet
from geom.poincare import expmap0, logmap

def seed_everything(seed : int) -> None: 
    '''
        Seed everything for reproducibility.
        Args:
            seed : any integer
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_scores_one_cluster(ftrain, ftest, food, shrunkcov=False):
    if shrunkcov:
        print("Using ledoit-wolf covariance estimator.")
        cov = lambda x: ledoit_wolf(x)[0]
    else:
        cov = lambda x: np.cov(x.T, bias=True)

    # ToDO: Simplify these equations
    dtest = np.sum(
        (ftest - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    dood = np.sum(
        (food - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (food - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    return dtest, dood

def get_scores_multi_cluster(ftrain, ftest, food, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]
    dood = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (food - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    din = np.min(din, axis=0)
    dood = np.min(dood, axis=0)

    return din, dood


def get_scores(ftrain, ftest, food, labelstrain, clusters):
    if clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, food)
    else:
        ypred = labelstrain
        return get_scores_multi_cluster(ftrain, ftest, food, ypred)

def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred


def get_eval_results(ftrain, ftest, food, labelstrain, clusters):
    """
    None.
    """
    dtest, dood = get_scores(ftrain, ftest, food, labelstrain, clusters)

    fpr95 = get_fpr(dtest, dood)
    auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
    return fpr95, auroc, aupr


def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr


def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)


#### Utils from KNN-OOD
def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = skm.roc_auc_score(labels, examples)
    aupr = skm.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def get_mean_prec(train_features, train_labels, n_cls):
    '''
    used for Mahalanobis score. Calculate class-wise mean and inverse covariance matrix
    '''
    classwise_mean = torch.empty(n_cls, train_features.shape[1],  device = 'cuda')
    classwise_cov = torch.empty(n_cls, train_features.shape[1], train_features.shape[1],  device = 'cuda')
    all_features = torch.zeros((0, train_features.shape[1]), device = 'cuda')
    classwise_idx = {} 
    all_features = train_features
    
    targets = train_labels
    for class_id in range(n_cls):
        classwise_idx[class_id] = np.where(targets == class_id)[0]
    
    for cls in range(n_cls):
        classwise_mean[cls] = torch.mean(all_features[classwise_idx[cls]].float(), dim = 0)
        classwise_cov[cls] = torch.cov(all_features[classwise_idx[cls]].float().T)
        
    tied_cov = torch.sum(classwise_cov,dim=0) / train_features.shape[1]
    precision = torch.linalg.pinv(tied_cov).float()
    precision = precision.to(classwise_mean.device)
    return classwise_mean, precision

def get_Mahalanobis_score(test_features, n_cls, classwise_mean, precision, in_dist = True):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    '''
    Mahalanobis_score_all = []
    with torch.no_grad():
        for i in range(n_cls):
            class_mean = classwise_mean[i]
            zero_f = test_features - class_mean
            Mahalanobis_dist = -torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
            if i == 0:
                Mahalanobis_score = Mahalanobis_dist.view(-1,1)
            else:
                Mahalanobis_score = torch.cat((Mahalanobis_score, Mahalanobis_dist.view(-1,1)), 1)      
        Mahalanobis_score, _ = torch.max(Mahalanobis_score, dim=1)
        Mahalanobis_score_all.extend(-Mahalanobis_score.cpu().numpy())
        
    return np.asarray(Mahalanobis_score_all, dtype=np.float32)


def exponential_map(x, c=1.0):
    """
    Exponential map for the Poincare ball model.

    Parameters:
    x (np.ndarray): Input vector in Euclidean space.
    c (float): Curvature parameter of hyperbolic space.

    Returns:
    np.ndarray: Mapped vector in hyperbolic space.
    """
    norm_x = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.tanh(np.sqrt(c) * norm_x) * x / (np.sqrt(c) * norm_x)

def embed_in_poincare_ball(X, c=1.0):
    """
    Embeds points in the Poincare ball.

    Parameters:
    X (np.ndarray): [N x d] dimensional matrix of Euclidean vectors.
    c (float): Curvature parameter of hyperbolic space.

    Returns:
    np.ndarray: [N x d] dimensional matrix of vectors in hyperbolic space.
    """
    return np.array([exponential_map(x, c=c) for x in X])

def distance_from_origin(Y):
    """
    Computes the distance from the origin for each point in hyperbolic space.

    Parameters:
    Y (np.ndarray): [N x d] dimensional matrix of vectors in hyperbolic space.

    Returns:
    np.ndarray: Distance of each point from the origin.
    """
    return np.arccosh(1 + 2 * np.sum(np.square(Y), axis=-1) / (1 - np.sum(np.square(Y), axis=-1)))

def distance_from_origin_stable(Y, epsilon=1e-7):
    """
    Computes the distance from the origin for each point in hyperbolic space, with numerical stability.

    Parameters:
    Y (np.ndarray): [N x d] dimensional matrix of vectors in hyperbolic space.
    epsilon (float): Small value to ensure numerical stability.

    Returns:
    np.ndarray: Distance of each point from the origin.
    """
    # Ensure the norm of Y is less than 1 to avoid numerical issues
    norm_Y = np.linalg.norm(Y, axis=-1)
    Y = Y / (1 + epsilon) * np.where(norm_Y >= 1, (1 - epsilon) / norm_Y, 1).reshape(-1, 1)

    # Compute the distance with an added epsilon for numerical stability
    return np.arccosh(1 + 2 * np.sum(np.square(Y), axis=-1) / ((1 - np.sum(np.square(Y), axis=-1)) + epsilon))


def clamp(value, eps=1e-10):
    if value >= 1:
        return 1 - eps
    else:
        return value

def mobius_addition(z, y, c):
    norm_y = np.linalg.norm(y)
    norm_z = np.linalg.norm(z)
    numerator = (1 + 2*c*np.dot(z, y) + c*norm_y**2)*z + (1 - c*norm_z**2)*y
    denominator = 1+2*c*np.dot(z,y) + c**2*norm_z**2*norm_y**2
    return numerator / denominator

def poincare_logarithm_map(z, y, c=1.0):
    zy = mobius_addition(-z, y, c)
    norm_zy = clamp(np.linalg.norm(zy))
    norm_z = clamp(np.linalg.norm(z))
    lambda_c = 2/(1 - c*norm_z**2)
    return 2/(np.sqrt(c)*lambda_c)*np.arctanh(np.sqrt(c)*norm_zy)*(zy/norm_zy)