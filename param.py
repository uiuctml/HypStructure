import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import math
import json

def adjust_learning_rate(hyper, optimizer, epoch):
    optim_param = hyper['optimizer']
    lr = optim_param['lr']
    eta_min = lr * (hyper["lr_decay_rate"] ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / hyper["epochs"])) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(hyper, epoch, batch_id, total_batches, optimizer):
    if hyper["warm"] and epoch <= hyper["warm_epochs"]:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (hyper["warm_epochs"] * total_batches)
        lr = hyper["warmup_from"] + p * (hyper["warmup_to"] - hyper["warmup_from"])

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def init_optim_schedule(model : nn.Module, params : dict, train_loader : DataLoader, exp_name : str, init_optimizer = None):
    optim_param = params['optimizer']
    if init_optimizer is None:
        optimizer = SGD(model.parameters(), **optim_param)
    else:
        optimizer = init_optimizer
        for p in optimizer.param_groups:
            p["lr"] = optim_param['lr']
            p["initial_lr"] = optim_param['lr']
    if exp_name.startswith('SupCon'):
        scheduler = None
    elif exp_name.startswith('ERM'):
        schedule_param = params['scheduler']
        schedule_param['step_size'] *= len(train_loader)
        scheduler = StepLR(optimizer, **schedule_param)
    return optimizer, scheduler

def load_params(dataset_name : str, exp_name : str) -> dict:
    reset = {}
    with open(f'./{dataset_name.lower()}/hyperparameters.json', 'r') as fp:
        params = json.load(fp)
    reset['epochs'] = params[exp_name]['epochs']
    reset['optimizer'] = params[exp_name]['optimizer']
    reset['lr_decay_rate'] = params[exp_name]['lr_decay_rate']
    if 'scheduler' in params[exp_name]:
        reset['scheduler'] = params[exp_name]['scheduler']
    else:
        reset['scheduler'] = None
    return reset