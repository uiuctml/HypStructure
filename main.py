import torch
from torch import Tensor 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import numpy as np
import math

import faiss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from datetime import datetime
import argparse
import os
import json
from typing import *
import pickle

from model import init_model
from data_loader import make_dataloader, TwoCropTransform
from loss import CPCCLoss, SupConLoss
from param import init_optim_schedule, adjust_learning_rate, warmup_learning_rate, load_params
from utils import *

import geom.pmath as pmath
from geom import poincare


def get_different_loss(exp_name : str, model : nn.Module, data : Tensor, 
                        criterion : nn.Module, target : Tensor, 
                        args) -> Tuple[Tensor, Tensor, Tensor]:
    '''
        Helper to calculate non CPCC loss, also return (default unnormalized) representation and model loss
    '''
    if exp_name == 'SupCon':
        bsz = target.shape[0]
        input_combined = torch.cat([data[0], data[1]], dim=0).cuda()
        target_combined = target.repeat(2).cuda()

        if isinstance(model, nn.DataParallel) or (hasattr(args, 'world_size') and args.world_size > 1): 
            model = model.module

        penultimate = model.encoder(input_combined).squeeze()
        representation = penultimate[:bsz]

        if args.normalize: # default: False or 0 
            penultimate = F.normalize(penultimate, dim=1)
        
        features = F.normalize(model.head(penultimate), dim=1) # result of proj head
        f1, f2 = torch.split(features, [bsz, bsz], dim=0) #f1 shape: [bz, feat_dim]
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) #features shape: [bz, 2, feat_dim]
        loss = criterion(features, target_combined[:bsz])
    
    elif exp_name == 'ERM':
        representation, logits = model(data)
        loss = criterion(logits, target)
    return representation, loss

def pretrain_objective(train_loader : DataLoader, val_loader : DataLoader, device : torch.device, 
               save_dir : str, seed : int, CPCC : bool, exp_name : str, epochs : int,
               dataset_name : str, hyper) -> None:
    '''
    Pretrain session.
    '''
    
    def trainer(model, hyper, total_epochs, optimizer, scheduler, train_loader, val_loader):
        train_losses_base_hist = {}
        train_losses_cpcc_hist = {}
        for epoch in range(1,total_epochs+1):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            adjust_learning_rate(hyper, optimizer, epoch)
            t_start = datetime.now() # record the time for each epoch
            model.train()
            
            train_losses_base = []
            train_losses_cpcc = []

            for idx, (data, target)  in enumerate(train_loader):
                warmup_learning_rate(hyper, epoch, idx, len(train_loader), optimizer)
                target = target.to(device)

                optimizer.zero_grad()

                representation, loss_base = get_different_loss(exp_name, model, data, criterion, target, args)

                loss_cpcc = criterion_cpcc(representation, target)
                train_losses_cpcc.append(loss_cpcc)
                train_losses_base.append(loss_base)
                if CPCC:
                    loss = loss_base + lamb * loss_cpcc
                    if args.center:
                        if args.cpcc_metric == 'poincare_mean':
                            if args.feature_dim < 512: # 64
                                loss = loss_base + lamb * 0.5 * loss_cpcc + 0.005 * torch.norm(pmath.poincare_mean(poincare.expmap0(representation),dim=0, c=1.0))
                            else:
                                loss = loss_base + lamb * loss_cpcc + 0.01 * torch.norm(pmath.poincare_mean(poincare.expmap0(representation),dim=0, c=1.0))
                        else:
                            loss = loss_base + lamb * loss_cpcc + 0.01 * torch.norm(torch.mean(representation,0))
                else:
                    loss = loss_base
                
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                if is_rank_zero() and (idx % 10 == 1):
                    print(f"Train Loss: {sum(train_losses_base)/len(train_losses_base):.4f}, "
                    f"Train CPCC: {sum(train_losses_cpcc)/len(train_losses_cpcc):.4f}")
            

            t_end = datetime.now()
            t_delta = (t_end-t_start).total_seconds()
            
            if is_rank_zero():
                print(f"Epoch {epoch} takes {t_delta} sec.")

                train_losses_base_hist[epoch] = (sum(train_losses_base)/len(train_losses_base)).item()
                train_losses_cpcc_hist[epoch] = (sum(train_losses_cpcc)/len(train_losses_cpcc)).item()

                pickle.dump(train_losses_base_hist, open(save_dir + f'/train_losses_base_hist.pkl', 'wb'))
                pickle.dump(train_losses_cpcc_hist, open(save_dir + f'/train_losses_cpcc_hist.pkl', 'wb'))


                log_dict = {f"train_losses_{exp_name}":sum(train_losses_base)/len(train_losses_base),
                            f"train_losses_cpcc":sum(train_losses_cpcc)/len(train_losses_cpcc),}


                if epoch % args.save_freq == 0 and args.save_freq > 0: 
                    checkpoint = {'epoch': epoch,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict()}
                    if scheduler is not None:
                        checkpoint['lr_sched'] = scheduler
                    torch.save(checkpoint, save_dir+f"/checkpoints/e{epoch}_seed{seed}.pth")

        return
    
    def base_eval(model, val_loader):
        model.eval() 
        test_accs = []
        test_losses_base = []
        test_losses_cpcc = []
        
        with torch.no_grad():
            for item in val_loader:
                data = item[0].to(device)
                target = item[-1].to(device)

                representation, logits = model(data)
                loss_base = criterion(logits, target)
                loss_cpcc = criterion_cpcc(representation, target)
                
                prob = F.softmax(logits,dim=1)
                pred = prob.argmax(dim=1)
                acc = pred.eq(target).flatten().tolist()
                test_accs.extend(acc)

                test_losses_base.append(loss_base)
                test_losses_cpcc.append(loss_cpcc)

        return sum(test_accs)/len(test_accs), sum(test_losses_base)/len(test_losses_base), sum(test_losses_cpcc)/len(test_losses_cpcc)

    def knn(model, val_loader):
        model.eval()

        features = []
        labels = []

        test_losses_base = []
        test_losses_cpcc = []

        with torch.no_grad():
            for item in val_loader:
                data = item[0]
                target = item[-1].to(device)
                bsz = target.shape[0]
                target_combined = target.repeat(2)

                # compute output
                input_combined = torch.cat([data[0], data[1]], dim=0).cuda()
                penultimate = model.module.encoder(input_combined).squeeze()

                representation = penultimate[:bsz]
                output = F.normalize(representation, dim=1).data.cpu()
                features.append(output)
                labels.append(target)

                proj_features = F.normalize(model.module.head(penultimate), dim=1) 
                f1, f2 = torch.split(proj_features, [bsz, bsz], dim=0) 
                proj_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) #
                supcon_loss = criterion(proj_features, target_combined[:bsz])
                test_losses_base.append(supcon_loss)
                
                representation, _ = model(data[0].to(device))
                test_loss_cpcc = criterion_cpcc(representation, target)
                test_losses_cpcc.append(test_loss_cpcc)

            features = torch.cat(features).numpy()
            labels = torch.cat(labels).cpu().numpy()

            cls = KNeighborsClassifier(20, metric="cosine").fit(features, labels)
            acc = np.mean(cross_val_score(cls, features, labels))


        return acc, sum(test_losses_base)/len(test_losses_base), sum(test_losses_cpcc)/len(test_losses_cpcc)
    
    if 'WORLD_SIZE' in os.environ and world_size > 0:
        dist.init_process_group()
    torch.set_float32_matmul_precision('high')

    dataset = train_loader.dataset
    num_train_batches = len(train_loader.dataset) // train_loader.batch_size + 1

    optim_config, scheduler_config = hyper['optimizer'], hyper['scheduler']
    init_config = {"dataset":dataset_name,
                "exp_name":exp_name,
                "cpcc":CPCC,
                "_batch_size":train_loader.batch_size,
                "epochs":epochs,
                "_num_workers":train_loader.num_workers,
                "cpcc_lamb":lamb,
                "center":args.center,
                }
    if CPCC:
        init_config['cpcc_metric'] = args.cpcc_metric
        init_config['leaf_only'] = args.leaf_only
        if args.cpcc_metric == 'poincare_exp_c':
            init_config['hyp_c'] = args.hyp_c
            init_config['clip_r'] = args.clip_r
            init_config['poincare_head_dim'] = args.poincare_head_dim
            init_config['train_c'] = args.train_c
    
    if scheduler_config is None:
        config = {**init_config, **optim_config}      
    else:
        config = {**init_config, **optim_config, **scheduler_config} 

    if exp_name.startswith('SupCon'):
        criterion = SupConLoss(temperature = temperature)
    elif exp_name.startswith('ERM'):
        criterion = nn.CrossEntropyLoss()
    
    criterion_cpcc = CPCCLoss(dataset, args)
    
    if is_rank_zero():
        with open(save_dir+'/config.json', 'w') as fp:
            json.dump(config, fp, sort_keys=True, indent=4)
    
    out_dir = save_dir+f"/seed{seed}.pth"
    if os.path.exists(out_dir):
        print("Skipped.")
        return
    
    model = init_model(device, args)

    if exp_name.startswith('SupCon'):
        # apply two crop to validation dataset to get the correct loss
        val_loader.dataset.transform = TwoCropTransform(val_loader.dataset.transform)

    if train_loader.batch_size > 256 and (exp_name.startswith('SupCon')) :
        hyper["warm"] = True
        hyper["warmup_from"] = 0.01
        hyper["warm_epochs"] = 10
        optim_param = hyper['optimizer']
        lr_decay_rate = hyper["lr_decay_rate"]
        eta_min = optim_param['lr'] * (lr_decay_rate ** 3)
        hyper["warmup_to"] = eta_min + (optim_param['lr'] - eta_min) * (
                    1 + math.cos(math.pi * hyper["warm_epochs"] / epochs)) / 2

    optimizer, scheduler = init_optim_schedule(model, hyper, train_loader, exp_name, init_optimizer=None)
    trainer(model, hyper, epochs, optimizer, scheduler, train_loader, val_loader)

    if is_rank_zero():
        torch.save(model.state_dict(), out_dir) # save the last checkpoint by convention
    
    # wandb.finish()
    if 'WORLD_SIZE' in os.environ and world_size > 0:
        ddp_cleanup()
    return

def feature_extractor(dataloader : DataLoader, seed : int, epoch : int = -1):
    model = init_model(device, args)

    model_dict = model.state_dict()
    if epoch == -1:
        ckpt_dict = {k: v for k, v in torch.load(save_dir+f"/seed{seed}.pth").items()}
    else:
        ckpt_dict = {k: v for k, v in torch.load(save_dir+f"/e{epoch}_seed{seed}.pth").items()}
    model_dict.update(ckpt_dict) 
    model.load_state_dict(model_dict)

    features = []
    targets = []
    model.eval()

    if isinstance(model, nn.DataParallel) or (hasattr(args, 'world_size') and args.world_size > 1): 
        model = model.module

    with torch.no_grad():
        for item in dataloader:
            data = item[0]
            target = item[-1]
                
            if isinstance(data, List):
                input_combined = data[0].cuda()
            else: # test dataset, type(data) = Tensor
                input_combined = data.cuda()

            penultimate = model.encoder(input_combined).squeeze()
            features.append(penultimate.cpu().detach().numpy())

            target = target.to(device)
            targets.append(target.cpu().detach().numpy())

    features = np.concatenate(features,axis=0)
    targets = np.concatenate(targets,axis=0)
    return features, targets

def ood_detection(seeds : int, in_dataset_name : str, ood_dataset_name : str, 
                  exp_name : str, num_workers : int, batch_size : int):

    def load_ood_scores(ood_dataset_name, method_name, out):
        print(f'{method_name} Skipped.')
        out[method_name]['unnormalized']['fpr95'].append(result[ood_dataset_name][method_name]['unnormalized']['fpr95'])
        out[method_name]['unnormalized']['auroc'].append(result[ood_dataset_name][method_name]['unnormalized']['auroc'])
        out[method_name]['unnormalized']['aupr'].append(result[ood_dataset_name][method_name]['unnormalized']['aupr'])
        out[method_name]['normalized']['fpr95'].append(result[ood_dataset_name][method_name]['normalized']['fpr95'])
        out[method_name]['normalized']['auroc'].append(result[ood_dataset_name][method_name]['normalized']['auroc'])
        out[method_name]['normalized']['aupr'].append(result[ood_dataset_name][method_name]['normalized']['aupr'])
        return out
    
    in_train_loader = make_dataloader(exp_name, num_workers, batch_size, 'train', in_dataset_name)
    in_test_loader = make_dataloader(exp_name, num_workers, batch_size, 'test', in_dataset_name)
    out_test_loader = make_dataloader(exp_name, num_workers, batch_size, 'ood', in_dataset_name, ood_dataset_name = ood_dataset_name)
    
    print("OOD Dataset:", ood_dataset_name)
    out = {'Mahalanobis':{'normalized' : {'fpr95':[],'auroc':[],'aupr':[]}, 'unnormalized' : {'fpr95':[],'auroc':[],'aupr':[]}},
             'SSD':{'normalized' : {'fpr95':[],'auroc':[],'aupr':[]}, 'unnormalized' : {'fpr95':[],'auroc':[],'aupr':[]}},
             'KNN':{'normalized' : {'fpr95':[],'auroc':[],'aupr':[]}, 'unnormalized' : {'fpr95':[],'auroc':[],'aupr':[]}},
          } 
    
    if os.path.exists(save_dir + '/OOD.json'):
        with open(save_dir+'/OOD.json', 'r') as fp:
            result = json.load(fp)
    else:
        result = dict()

    for seed in range(seeds):
        # compute features
        save_location = save_dir + '/' + ood_dataset_name
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        if not os.path.exists(save_location + f'/in_train_features_seed{seed}.pkl') or \
            not os.path.exists(save_location + f'/in_train_labels_seed{seed}.pkl'):
            in_train_features, in_train_labels = feature_extractor(in_train_loader, seed)
            pickle.dump(in_train_features, open(save_location + f'/in_train_features_seed{seed}.pkl', 'wb'))
            pickle.dump(in_train_labels, open(save_location + f'/in_train_labels_seed{seed}.pkl', 'wb'))
        else:
            with open(save_location + f'/in_train_features_seed{seed}.pkl', 'rb') as fp:
                in_train_features = pickle.load(fp)
            with open(save_location + f'/in_train_labels_seed{seed}.pkl', 'rb') as fp:
                in_train_labels = pickle.load(fp)
        if not os.path.exists(save_location + f'/in_test_features_seed{seed}.pkl') or \
            not os.path.exists(save_location + f'/in_test_labels_seed{seed}.pkl'):
            in_test_features, in_test_labels = feature_extractor(in_test_loader, seed)
            pickle.dump(in_test_features, open(save_location + f'/in_test_features_seed{seed}.pkl', 'wb'))
            pickle.dump(in_test_labels, open(save_location + f'/in_test_labels_seed{seed}.pkl', 'wb'))
        else:
            with open(save_location + f'/in_test_features_seed{seed}.pkl', 'rb') as fp:
                in_test_features = pickle.load(fp)
            with open(save_location + f'/in_test_labels_seed{seed}.pkl', 'rb') as fp:
                in_test_labels = pickle.load(fp)
        if not os.path.exists(save_location + f'/out_test_features_seed{seed}.pkl') or \
            not os.path.exists(save_location + f'/out_test_labels_seed{seed}.pkl'):
            out_test_features, out_test_labels = feature_extractor(out_test_loader, seed)
            pickle.dump(out_test_features, open(save_location + f'/out_test_features_seed{seed}.pkl', 'wb'))
            pickle.dump(out_test_labels, open(save_location + f'/out_test_labels_seed{seed}.pkl', 'wb'))
        else:
            with open(save_location + f'/out_test_features_seed{seed}.pkl', 'rb') as fp:
                out_test_features = pickle.load(fp)
            with open(save_location + f'/out_test_labels_seed{seed}.pkl', 'rb') as fp:
                out_test_labels = pickle.load(fp)
        print("Features successfully loaded.")


        ftrain = np.copy(in_train_features)
        ftest = np.copy(in_test_features)
        food = np.copy(out_test_features)
        ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
        ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
        food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

        m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

        ftrain = (ftrain - m) / (s + 1e-10)
        ftest = (ftest - m) / (s + 1e-10)
        food = (food - m) / (s + 1e-10)

        clusters = 1
        fpr95_s, auroc_s, aupr_s = get_eval_results(
            np.copy(in_train_features),
            np.copy(in_test_features),
            np.copy(out_test_features),
            np.copy(in_train_labels),
            clusters,
        )
        print("SSD: FPR95", fpr95_s, "AUROC:", auroc_s, "AUPR:", aupr_s) 
        out['SSD']['unnormalized']['fpr95'].append(fpr95_s)
        out['SSD']['unnormalized']['auroc'].append(auroc_s)
        out['SSD']['unnormalized']['aupr'].append(aupr_s)
    
        clusters = 1
        fpr95_sn, auroc_sn, aupr_sn = get_eval_results(
            np.copy(ftrain),
            np.copy(ftest),
            np.copy(food),
            np.copy(in_train_labels),
            clusters,
        )
        print("Normalized SSD: FPR95", fpr95_sn, "AUROC:", auroc_sn, "AUPR:", aupr_sn)
        out['SSD']['normalized']['fpr95'].append(fpr95_sn)
        out['SSD']['normalized']['auroc'].append(auroc_sn)
        out['SSD']['normalized']['aupr'].append(aupr_sn)

        if in_dataset_name == 'CIFAR10':
            K = 50
        elif in_dataset_name == 'CIFAR100':
            K = 200
        elif in_dataset_name == 'IMAGENET100':
            K = 200
        index = faiss.IndexFlatL2(in_train_features.shape[1])
        index.add(in_train_features)
        D, _ = index.search(in_test_features, K)
        in_score = D[:,-1]

        D, _ = index.search(out_test_features,K)
        out_score = D[:,-1]

        auroc_k, aupr_k, fpr_k = get_measures(-in_score, -out_score) 
        print("KNN: FPR95", fpr_k, "AUROC:", auroc_k, "AUPR:", aupr_k)
        out['KNN']['unnormalized']['fpr95'].append(fpr_k)
        out['KNN']['unnormalized']['auroc'].append(auroc_k)
        out['KNN']['unnormalized']['aupr'].append(aupr_k)

        index = faiss.IndexFlatL2(ftrain.shape[1])
        index.add(ftrain)
        D, _ = index.search(ftest, K)
        in_score = D[:,-1]

        D, _ = index.search(food,K)
        out_score = D[:,-1]

        auroc_kn, aupr_kn, fpr_kn = get_measures(-in_score, -out_score) 
        print("Normalized KNN: FPR95", fpr_kn, "AUROC:", auroc_kn, "AUPR:", aupr_kn)
        out['KNN']['normalized']['fpr95'].append(fpr_kn)
        out['KNN']['normalized']['auroc'].append(auroc_kn)
        out['KNN']['normalized']['aupr'].append(aupr_kn)
        
        n_cls = args.n_cls
        classwise_mean, precision = get_mean_prec(torch.tensor(in_train_features), torch.tensor(in_train_labels), n_cls)
        in_score_maha = get_Mahalanobis_score(torch.tensor(in_test_features).to(device), n_cls, classwise_mean, precision, in_dist = True)
        out_score_maha = get_Mahalanobis_score(torch.tensor(out_test_features).to(device), n_cls, classwise_mean, precision, in_dist = False)
        auroc_m, aupr_m, fpr_m = get_measures(-in_score_maha, -out_score_maha) 
        print("Mahalanobis: FPR95", fpr_m, "AUROC:", auroc_m, "AUPR:", aupr_m)
        out['Mahalanobis']['unnormalized']['fpr95'].append(fpr_m)
        out['Mahalanobis']['unnormalized']['auroc'].append(auroc_m)
        out['Mahalanobis']['unnormalized']['aupr'].append(aupr_m)

        n_cls = args.n_cls
        classwise_mean, precision = get_mean_prec(torch.tensor(ftrain), torch.tensor(in_train_labels), n_cls)
        in_score_maha = get_Mahalanobis_score(torch.tensor(ftest).to(device), n_cls, classwise_mean, precision, in_dist = True)
        out_score_maha = get_Mahalanobis_score(torch.tensor(food).to(device), n_cls, classwise_mean, precision, in_dist = False)
        auroc_mn, aupr_mn, fpr_mn = get_measures(-in_score_maha, -out_score_maha) 
        print("Normalized Mahalanobis: FPR95", fpr_mn, "AUROC:", auroc_mn, "AUPR:", aupr_mn)
        out['Mahalanobis']['normalized']['fpr95'].append(fpr_mn)
        out['Mahalanobis']['normalized']['auroc'].append(auroc_mn)
        out['Mahalanobis']['normalized']['aupr'].append(aupr_mn)

    for ood_scores in ['Mahalanobis','SSD','KNN']:
        for n in ['normalized','unnormalized']:
            for metric in ['fpr95','auroc','aupr']:
                out[ood_scores][n][metric] = np.mean(out[ood_scores][n][metric]) 
    
    result[ood_dataset_name] = out
    with open(save_dir+'/OOD.json', 'w') as fp:
        json.dump(result, fp, indent=4)
    
    return result

def main():
    
    # Train
    for seed in range(seeds):
        seed_everything(seed)
        hyper = load_params(dataset_name, exp_name)
        epochs = hyper['epochs']
        train_loader = make_dataloader(exp_name, num_workers, batch_size, 'train', dataset_name)
        val_loader = make_dataloader(exp_name, num_workers, batch_size, 'test', dataset_name)
        args.n_cls = len(train_loader.dataset.leaf_names)
        pretrain_objective(train_loader, val_loader, device, save_dir, seed, cpcc, exp_name, epochs, dataset_name, hyper)

    # Eval: ood
    if dataset_name in ['CIFAR100','CIFAR10']:
        ood_dataset_names = ['SVHN', 'Textures', 'Places365', 'LSUN','iSUN']
    elif dataset_name in ['IMAGENET100']:
        ood_dataset_names = ['iNaturalist','SUN', 'Places365', 'dtd']
    for ood_dataset_name in ood_dataset_names: 
        ood_detection(seeds, dataset_name, ood_dataset_name, exp_name, num_workers, batch_size)
    
    return

def is_ddp_initialized():
    return torch.distributed.is_initialized()

def is_rank_zero():
    # Check if this is the rank 0 process or if DDP is not initialized (i.e., single GPU/CPU mode)
    if is_ddp_initialized():
        return torch.distributed.get_rank() == 0
    return True


def ddp_setup():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group()

    device_id = rank
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device_id) 

    return device, device_id, rank, world_size

def ddp_cleanup():
    dist.destroy_process_group()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/path/to/data", type=str, help='directory that you want to save your experiment results')
    parser.add_argument("--timestamp", required=True, help=r'your unique experiment id, hint: datetime.now().strftime("%m%d%Y%H%M%S")') 
    parser.add_argument("--exp_name", required=True, help='ERM | SupCon')
    parser.add_argument("--dataset", required=True, help='CIFAR100 | CIFAR10 | IMAGENET100')
    parser.add_argument("--model_name", type=str, help='resnet18 | resnet34 | resnet50')
    parser.add_argument("--save_freq", type=int, default=-1)
    
    parser.add_argument("--cpcc", required=True, type=int, help='0/1')
    parser.add_argument("--cpcc_metric", default='poincare', type=str, help='distance metric in CPCC, l2/l1/poincare')
    parser.add_argument("--leaf_only", default=0, type=int, help='0 use all nodes, 1 use leaf nodes')
    parser.add_argument("--lamb",type=float,default=1,help='strength of CPCC regularization')
    parser.add_argument("--center", default=0, type=int, help='normalize batch representation')
    parser.add_argument("--poincare_low_dim_project", type=int, help='0/1')
    parser.add_argument("--poincare_head_dim", type=int, default=512)
    parser.add_argument("--hyp_c",type=float,default=1.0,help='curvature of the poincare ball')
    parser.add_argument("--clip_r",type=float,default=2.3,help='Clipping parameter of the poincare ball')
    parser.add_argument("--train_c",type=bool,default=False,help='Whether or not to train the curvature')
    parser.add_argument("--feature_dim", type=int, help='change feature dimension of encoder')

    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seeds", type=int,default=5)
    
    parser.add_argument('--warm', action='store_true',
                            help='warm-up for large batch training')
    
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
    
    parser.add_argument('--normalize', type=int, default=0,
                        help='normalize feat embeddings setting, see comments in loss.py')
    
    parser.set_defaults(bottleneck=True)
    parser.set_defaults(augment=True)

    parser.add_argument("--local-rank", type=int, default=0) # DDP
    parser.add_argument('--encoder_manifold', type=str, choices=['euclidean','poincare'])

    
    args = parser.parse_args()
    timestamp = args.timestamp
    exp_name = args.exp_name
    dataset_name = args.dataset
    cpcc = args.cpcc

    num_workers = args.num_workers
    batch_size = args.batch_size
    seeds = args.seeds
    lamb = args.lamb

    temperature = args.temp

    root = f'{args.root}/hypstructure/{dataset_name}' 
    save_dir = root + '/' + timestamp 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/checkpoints')


    # Check if the script is running in a DDP environment
    if 'WORLD_SIZE' in os.environ:
        # Assuming DDP is initialized outside this function if required
        world_size = int(os.environ.get('WORLD_SIZE',0))
        args.world_size = world_size
        if world_size > 1:  # More than one process implies DDP
            local_rank = int(os.getenv('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            rank = int(os.environ['RANK'])
            args.rank = rank
            print(f"Running in DDP mode on device: {device}, rank: {rank}/{world_size}")
    else:
        # Default to single GPU/CPU training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running in single GPU/CPU mode on device: {device}")
    
    main()
