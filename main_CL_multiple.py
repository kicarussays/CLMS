import argparse
import logging
import traceback
import os
import copy
import pickle
import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import DataLoader

from src.utils import LoadData, useDataset, sitesum, seedset, aucfloat
from src.models import ResNet1d
from src.learning import Supervised_Learning, Federated_Learning, Continual_Learning
from src.config import today, seed

torch.set_num_threads(16)

parser = argparse.ArgumentParser(description='classifier Training')
parser.add_argument('--device', '-d', type=int, 
                    help='GPU Device Number', default=0)
parser.add_argument('--mode', '-m', type=str, 
                    help='Supervised or FedAvg or FedProx or Finetuning or Continual')
parser.add_argument('--bs', type=int, 
                    help='Batch Size', default=256)
parser.add_argument('--lr', type=float, 
                    help='Learning Rate', default=0.0001)
parser.add_argument('--round', '-r', type=int, 
                    help='Iteration Round for Federated Learning', default=30)
parser.add_argument('--mu', type=float, 
                    help='hyperparameter for FedProx', default=0.01)
parser.add_argument('--epoch', '-e', type=int, 
                    help='Max Epochs for Each Site', default=100)
parser.add_argument('--valtime', '-v', type=int, 
                    help='Validation Interval', default=1)
parser.add_argument('--patience', '-p', type=int, 
                    help='Early Stopping Patience', default=10)
parser.add_argument('--es', type=int, 
                    help='early stopping option', default=1)
parser.add_argument('--seed', type=int, 
                    help='Random Seed', default=678)
parser.add_argument('--alpha', type=float, 
                    help='Hyperparameter decaying factor for continual learning methods', default=0.9)
parser.add_argument('--delta', type=float, 
                    help='Performance drop margin', default=0.05)
parser.add_argument('--domain', type=str, 
                    help='Domain', default='multiple')
parser.add_argument('--order', type=str, 
                    help='Small to Large or Large to Small (stl, lts)', default='eth')




def main():
    args = parser.parse_args()
    seedset(args.seed)
    
    
    
    
    if args.device != 'cpu':
        device = f'cuda:{args.device}'
    else:
        device = args.device
        
    if args.mode == 'Supervised':
        params = f'{today}_{args.mode}_{args.bs}_{args.lr}_{args.epoch}_{args.seed}'
    elif args.mode in ('FedAvg', 'FedProx'):
        params = f'{today}_{args.mode}_{args.bs}_{args.lr}_{args.epoch}_{args.round}_{args.seed}'
    elif args.mode in ('Finetuning'):
        params = f'{today}_{args.mode}_{args.order}_{args.bs}_{args.lr}_{args.epoch}_{args.seed}'
    elif args.mode in ('Continual'):
        params = f'{today}_{args.mode}_{args.order}_{args.bs}_{args.lr}_{args.alpha}_{args.delta}_{args.sub_epoch}_{args.seed}'
        
        
    savepath = f'results/{args.domain}/{args.mode}/saved/{params}/'
    logpath = f'results/{args.domain}/{args.mode}/logs/'
    os.makedirs(savepath, exist_ok=True)
    os.makedirs(logpath, exist_ok=True)
        
    logging.basicConfig(filename=os.path.join(logpath, f'{params}.log'), 
                level=logging.DEBUG,
                format='%(asctime)s:%(levelname)s:%(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True
    
    print("Data Loading Start")
    train_site_data, local_site_data, global_site_data = LoadData(seed)    
    
    print("Done")
    
    
    if args.mode == 'Supervised':
        for g, trainset in train_site_data.items():
            # Each site training
            model = ResNet1d()
            savepath = f'results/{args.domain}/{args.mode}/saved/{params}/{g}/'
            os.makedirs(savepath, exist_ok=True)
            
            logging.debug(f'\n\n{g} training...')
            Trainer = Supervised_Learning(
                train_loader = DataLoader(useDataset(trainset), batch_size=args.bs, shuffle=True),
                val_loader = DataLoader(useDataset(local_site_data[g]), batch_size=args.bs, shuffle=False),
                model = model,
                savepath = savepath,
                args = args
            )
            
            # Train (return best model)
            load_model, _ = Trainer.train()
            model.load_state_dict(load_model)
            
            # Test
            loss, roc_auc, pr_auc = Trainer.evaluation(model, DataLoader(useDataset(global_site_data[g]), batch_size=args.bs, shuffle=False))
            logging.debug("Final Test Score\nLoss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}".format(loss, roc_auc, pr_auc))
            
        
        # All site training
        logging.debug(f'\n\nAll training...')
        model = ResNet1d()
        savepath = f'results/{args.domain}/{args.mode}/saved/{params}/all/'
        Trainer = Supervised_Learning(
            train_loader = DataLoader(useDataset(sitesum(train_site_data)), batch_size=args.bs, shuffle=True),
            val_loader = DataLoader(useDataset(sitesum(local_site_data)), batch_size=args.bs, shuffle=False),
            model = model,
            savepath = savepath,
            args = args,
            singlelast=True
        )
        
        # Train (return best model)
        load_model, _ = Trainer.train()
        model.load_state_dict(load_model)
        
        logging.debug("Final Test Score\n")
        for site in global_site_data.keys():
            loss, roc_auc, pr_auc = Trainer.evaluation(model, DataLoader(useDataset(global_site_data[site]), batch_size=args.bs, shuffle=False))
            logging.debug("Site {} Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}".format(site, loss, roc_auc, pr_auc))
            
        loss, roc_auc, pr_auc = Trainer.evaluation(model, DataLoader(useDataset(sitesum(global_site_data)), batch_size=args.bs, shuffle=False))
        logging.debug("Site All Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}".format(loss, roc_auc, pr_auc))
        
    
    
    elif args.mode in ('FedAvg', 'FedProx'):
        model = ResNet1d()
        
        Trainer = Federated_Learning(
            train_site_data = train_site_data,
            local_site_data = local_site_data,
            model = model,
            savepath = savepath,
            args = args
        )
        
        best_global_model, best_local_model = Trainer.train()
        
        logging.debug("\n\nFinal Test AUROC:")
        for site in global_site_data.keys():
            _weight = model.state_dict()
            _weight.update(best_global_model)
            model.load_state_dict(_weight)
            loss, roc_auc, pr_auc = Trainer.evaluation(model.to(device), DataLoader(useDataset(global_site_data[site]), batch_size=args.bs, shuffle=False))
            logging.debug("Site [{}] AUROC: {:.4f}, AUPRC: {:.4f}".format(site, roc_auc, pr_auc))
            
    
    
    elif args.mode in ('Finetuning', 'Continual'):
        logpth = f'results/multiple/Supervised/logs/'
        useres = os.listdir(logpth)
        
        scores = []
        for u in useres:
            if str(args.seed) not in u: continue
            with open(os.path.join(logpth, u), "r") as f:
                ex = f.readlines()
            
            sidx = [_n for _n, e in enumerate(ex) if 'Final Test Score' in e]
            scores = []
            for n, idx in enumerate(sidx):
                if n < len(sidx) - 1:
                    scores.append(aucfloat(ex[idx+1].split(' ')[-3][:-1]))
                    
        
        Ns = [len(v) for k, v in train_site_data.items()]
        sorder = ['shaoxing', 'ptb', 'georgia', 'cpsc']
        if args.order == 'stl':
            score_rank = list(
                pd.DataFrame(
                    Ns, columns=['score'], index=sorder).sort_values(
                        by='score', ascending=True).index)
        elif args.order == 'lts':
            score_rank = list(
                pd.DataFrame(
                    Ns, columns=['score'], index=sorder).sort_values(
                        by='score', ascending=False).index)
        
        train_site_data = {s: train_site_data[s] for s in score_rank}
        local_site_data = {s: local_site_data[s] for s in score_rank}
        global_site_data = {s: global_site_data[s] for s in score_rank}
            
        
        model = ResNet1d()
        
        Trainer = Continual_Learning(
            train_site_data = train_site_data,
            local_site_data = local_site_data,
            model = model,
            savepath = savepath,
            args = args
        )
        
        best_global_model, best_local_model = Trainer.train()
        
        logging.debug("\n\nFinal Test AUROC:")
        for site in global_site_data.keys():
            _weight = model.state_dict()
            _weight.update(best_global_model)
            model.load_state_dict(_weight)
            loss, roc_auc, pr_auc = Trainer.evaluation(model.to(device), DataLoader(useDataset(global_site_data[site]), batch_size=args.bs, shuffle=False))
            logging.debug("Site [{}] AUROC: {:.4f}, AUPRC: {:.4f}".format(site, roc_auc, pr_auc))
    
    assert args.mode in ('Supervised', 'FedAvg', 'FedProx', 'Finetuning', 'Continual'), f'{args.mode} is not the case.'


if __name__ == "__main__":
    try:
        main()
        
    except:
        logging.error(traceback.format_exc())
    
    