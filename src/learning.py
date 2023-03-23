import os
import logging
import copy
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import torch
from torch.utils.data import DataLoader

from src.utils import LoadData, LoadSingleData, Components, useDataset, OmegaForMAS, FisherForEWC, LoadSynthesized
from src.process import LwF, MAS, EWC
from src.config import today, seed



class Supervised_Learning(Components):
    def __init__(self, train_loader, val_loader, model, savepath, args, weight=None, singlelast=False): 
        super().__init__()
        self.get_joint_components(train_loader, val_loader, model, savepath, args, weight)
        self.singlelast = singlelast
        
        
    def train(self):
        if not os.path.exists(self.savepath): os.mkdir(self.savepath)
        
        d = self.device
        bestauc = 0.0
        patience = 0
        if self.args.mode == 'Supervised' and self.args.domain == 'multiple' and self.singlelast:
            _, _local_site_data, _ = LoadData(seed)
        
        if self.args.mode == 'Supervised' and self.args.domain == 'single' and self.singlelast:
            _, _local_site_data, _ = LoadSingleData(seed, self.args.file)
                
        for e in range(self.args.epoch):
            self.model.train()
            loss_sum = 0
            skipcnt = 0
            
            for age, sex, ecg, flag in tqdm(self.train_loader):
                
                if ecg.shape[0] == 1:
                    skipcnt += 1
                    continue
                
                age, sex, ecg, flag = age.to(d), sex.to(d), ecg.to(d), flag.to(d)
                
                output = self.model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
                loss = self.criterion(output, flag.type(torch.long))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()

                
            if (e + 1) % self.args.valtime == 0:
                if self.singlelast:
                    for site, data in _local_site_data.items(): 
                        _val_loader = DataLoader(useDataset(data), batch_size=self.args.bs, shuffle=False)
                        val_loss, roc_auc, pr_auc = self.evaluation(self.model, _val_loader)
                        logging.debug('Epoch [{}/{}], Site [{}] Train Loss: {:.4f}, Val Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}'.format(
                                    e+1, self.args.epoch, site, loss_sum / (len(self.train_loader) - skipcnt), val_loss, roc_auc, pr_auc))
                
                else:
                    val_loss, roc_auc, pr_auc = self.evaluation(self.model, self.val_loader)
                    logging.debug('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}'.format(
                                e+1, self.args.epoch, loss_sum / (len(self.train_loader) - skipcnt), val_loss, roc_auc, pr_auc))
                
                if roc_auc > bestauc + 0.001:
                    logging.debug(f'Saved Best Model...')
                    torch.save(self.model.state_dict(), self.savepath + 'best_supervised_model.pt')
                    bestauc = roc_auc
                    patience = 0
                else:
                    patience += 1

            
            if self.args.es and patience > self.args.patience:
                logging.debug(f'Early Stopping Activated')
                break
        
        
        # torch.save(self.model.state_dict(), self.savepath + 'best_supervised_model.pt')
        load_model = torch.load(self.savepath + 'best_supervised_model.pt', map_location=self.device)
        
        return load_model, bestauc
        



class Federated_Learning(Components):
    def __init__(self, train_site_data, local_site_data, model, savepath, args):
        super().__init__()
        self.get_separated_components(train_site_data, local_site_data, model, savepath, args)
        
    def train(self):
        bestauc = 0.0
        patience = 0
        global_model = copy.deepcopy(self.model)
        
        all_weights = {site: None for site in self.train_site_data.keys()}
        global_trainer = {}
        local_trainer = {}
        
        for r in range(self.args.round):                
            logging.debug(f'\n\nRound {r+1}\n\n')
            
            
            for n, (site, data) in enumerate(self.train_site_data.items()):
                
                _use_model = copy.deepcopy(global_model)
                
                if self.args.mode != 'FedAvg' and r != 0:
                    _use_weight = _use_model.state_dict()
                    _use_weight.update(local_trainer[site])
                    _use_model.load_state_dict(_use_weight)
                    
                logging.debug(f'\n\n{site} training...\n\n')
                Trainer = Supervised_Learning(
                    train_loader=DataLoader(useDataset(data), batch_size=self.args.bs, shuffle=True),
                    val_loader=DataLoader(useDataset(self.local_site_data[site]), batch_size=self.args.bs, shuffle=False),
                    model=_use_model,
                    args=self.args,
                    savepath=self.savepath,
                    weight=all_weights[site]
                )
                
                if self.args.mode == 'FedAvg':
                    global_trainer[site], _ = Trainer.train()
                else:
                    assert self.args.mode not in ('FedAvg'), f"{self.args.mode} is not applicable"
                
                if r == 0: all_weights[site] = Trainer.weight
            
            
            avg_trainer = OrderedDict()
            for _site, _trainer in global_trainer.items(): break
            for k in _trainer.keys():
                for n, site in enumerate(global_trainer.keys()):
                    if n == 0:
                        avg_trainer[k] = global_trainer[site][k] * self.coefficients[site]
                    else:
                        avg_trainer[k] += global_trainer[site][k] * self.coefficients[site]
                
            
            macro_auc = 0
            total_cnt = 0
            for site, data in self.local_site_data.items():
                test_weights = global_model.state_dict()
                test_weights.update(avg_trainer)
                global_model.load_state_dict(test_weights)
                _testloader = DataLoader(useDataset(data), batch_size=self.args.bs, shuffle=False)
                loss, roc_auc, pr_auc = self.evaluation(global_model.to(self.device), _testloader)
                logging.debug('Round [{}/{}], Site [{}] AUROC: {:.4f}, AUPRC: {:.4f}'.format(
                            r+1, self.args.round, site, roc_auc, pr_auc))
                macro_auc += roc_auc * len(data)
                total_cnt += len(data)
                
            macro_auc = macro_auc / total_cnt
            
            if macro_auc > bestauc + 0.001:
                bestauc = macro_auc
                logging.debug(f'Saved Best Local Validation Model...')
                torch.save(global_model.state_dict(), self.savepath + 'best_global_model.pt')
                torch.save(local_trainer, self.savepath + 'best_local_model.pt')
                patience = 0
            
            else:
                patience += 1
            
                
            # if patience > self.args.patience:
            #     logging.debug(f'Early Stopping Activated')
            #     break
        
        load_global_model = torch.load(self.savepath + 'best_global_model.pt', map_location=self.device)
        load_local_model = torch.load(self.savepath + 'best_local_model.pt', map_location=self.device)
        
        return load_global_model, load_local_model




class Continual_Learning(Components):
    def __init__(self, train_site_data, local_site_data, model, savepath, args):
        super().__init__()
        self.get_separated_components(train_site_data, local_site_data, model, savepath, args)
        
    
    def train(self):
        d = self.device
        local_trainer = {}
        synthesized = []
        for n, (site, data) in enumerate(self.train_site_data.items()):
            saved_model_exists = False
            
            if n == 0:
                _use_model = copy.deepcopy(self.model)
                _use_model = _use_model.to(d)
                
                if self.args.domain == 'multiple':
                    saved_models = os.listdir(f"results/{self.args.domain}/Supervised/saved/")
                    for m in saved_models:
                        if str(self.args.seed) in m:
                            saved_model_exists = True
                            prev_path = f"results/{self.args.domain}/Supervised/saved/{m}/{site}/best_supervised_model.pt"
                            break
            
                elif self.args.domain == 'single':
                    saved_models = os.listdir(f"results/{self.args.domain}/Supervised/{self.args.file}/saved/")
                    for m in saved_models:
                        if str(self.args.seed) in m:
                            saved_model_exists = True
                            prev_path = f"results/{self.args.domain}/Supervised/{self.args.file}/saved/{m}/{site}/best_supervised_model.pt"
                            break
                        
                prev_weight = torch.load(prev_path, map_location=self.device)
                _use_model.load_state_dict(prev_weight)
                logging.debug('Site [{}] model already exists.'.format(site))
                local_trainer[site] = {k: v for k, v in prev_weight.items() if k.startswith(('fc', 'bn'))}
                _testloader = DataLoader(useDataset(self.local_site_data[site]), batch_size=self.args.bs, shuffle=False)
                loss, roc_auc, pr_auc = self.evaluation(_use_model, _testloader)
            
                logging.debug('Site [{}. {}], AUROC: {:.4f}, AUPRC: {:.4f}'.format(
                            n+1, site, roc_auc, pr_auc))
            
            
            if not saved_model_exists:
                init_weight = copy.deepcopy(_use_model.state_dict())
                # Finetuning
                logging.debug(f'\n\n{site} training...\n\n')
                logging.debug('Finetuning Start')
                Trainer = Supervised_Learning(
                    train_loader=DataLoader(useDataset(data), batch_size=self.args.bs, shuffle=True),
                    val_loader=DataLoader(useDataset(self.local_site_data[site]), batch_size=self.args.bs, shuffle=False),
                    model=_use_model,
                    savepath=self.savepath,
                    args=self.args,
                    weight=None
                )
                _get_weight, _get_auc = Trainer.train()
                auc_threshold = (1 - self.args.delta) * _get_auc
                if self.args.mode == 'Finetuning':
                    local_trainer[site] = {k: v for k, v in _get_weight.items() if k.startswith(('fc', 'bn'))}
                
                if n != 0:
                    continual_models = {}
                    if self.args.mode == 'Continual':
                        _use_model.load_state_dict(init_weight)
                        Trainer = LwF(
                            train_loader=DataLoader(useDataset(data), batch_size=self.args.bs, shuffle=True),
                            val_loader=DataLoader(useDataset(self.local_site_data[site]), batch_size=self.args.bs, shuffle=False),
                            model=_use_model,
                            savepath=self.savepath,
                            args=self.args,
                            weight=None
                        )
                        # _get_weight, _local_trainer, _sub_epoch = Trainer.train(local_trainer, site, auc_threshold)
                        # continual_models['LwF'] = (_get_weight, _local_trainer, _sub_epoch)
                        _get_weight, _local_trainer, _lambda = Trainer.train(local_trainer, site, auc_threshold)
                        continual_models['LwF'] = (_get_weight, _local_trainer, _lambda)
                        
                        _use_model.load_state_dict(init_weight)
                        Trainer = EWC(
                            train_loader=DataLoader(useDataset(data), batch_size=self.args.bs, shuffle=True),
                            val_loader=DataLoader(useDataset(self.local_site_data[site]), batch_size=self.args.bs, shuffle=False),
                            model=_use_model,
                            savepath=self.savepath,
                            args=self.args,
                            weight=None
                        )
                        # _get_weight, _local_trainer, _sub_epoch = Trainer.train(local_trainer, site, fisher, auc_threshold)
                        # continual_models['EWC'] = (_get_weight, _local_trainer, _sub_epoch)
                        _get_weight, _local_trainer, _lambda = Trainer.train(local_trainer, site, fisher, auc_threshold)
                        continual_models['EWC'] = (_get_weight, _local_trainer, _lambda)
                        
                        _use_model.load_state_dict(init_weight)
                        Trainer = MAS(
                            train_loader=DataLoader(useDataset(data), batch_size=self.args.bs, shuffle=True),
                            val_loader=DataLoader(useDataset(self.local_site_data[site]), batch_size=self.args.bs, shuffle=False),
                            model=_use_model,
                            savepath=self.savepath,
                            args=self.args,
                            weight=None
                        )
                        # _get_weight, _local_trainer, _sub_epoch = Trainer.train(local_trainer, site, omega, auc_threshold)
                        # continual_models['LwF'] = (_get_weight, _local_trainer, _sub_epoch)
                        _get_weight, _local_trainer, _lambda = Trainer.train(local_trainer, site, omega, auc_threshold)
                        continual_models['MAS'] = (_get_weight, _local_trainer, _lambda)
                        
                        all_reg_loss = []
                        bestauc = 0.0
                        for method, (test_weights, test_local_trainer, test_lambda) in continual_models.items():
                            _use_model.load_state_dict(test_weights)
                            _use_model.eval()
                            
                            synth_loader = DataLoader(useDataset(synthesized), batch_size=self.args.bs, shuffle=False)
                            all_flag = []
                            all_pred = []
                            for age, sex, ecg, flag in tqdm(synth_loader):
                                age, sex, ecg, flag = age.to(d), sex.to(d), ecg.to(d), flag.to(d)
                                _pred = _use_model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
                                all_flag.append(flag.cpu().detach().numpy())
                                all_pred.append(_pred[:, 1].cpu().detach().numpy())
    
                            all_flag, all_pred = np.concatenate(all_flag), np.concatenate(all_pred)

                            fpr, tpr, _ = roc_curve(all_flag, all_pred)
                            roc_auc = auc(fpr, tpr)
                            logging.debug(f'Method [{method}] : AUC {round(roc_auc, 4)}')
                            
                            if roc_auc > bestauc:
                                bestmethod = method
                                bestauc = roc_auc
                            
                        _get_weight, local_trainer, _ = continual_models[bestmethod]
                        logging.debug('\n\n\n[{}] Selected.\n\n\n'.format(bestmethod))
                        
                        
                
                _use_model.load_state_dict(_get_weight)
                for site, weights in local_trainer.items():
                    _testloader = DataLoader(useDataset(self.local_site_data[site]), batch_size=self.args.bs, shuffle=False)
                    loss, roc_auc, pr_auc = self.evaluation(_use_model, _testloader)
                                    
                    logging.debug('Site [{}. {}], AUROC: {:.4f}, AUPRC: {:.4f}'.format(
                                n+1, site, roc_auc, pr_auc))
            
            
            if self.args.mode == 'Continual':
                if n < len(self.train_site_data) - 1: 
                    # Fisher Information Matrix for EWC
                    print('Fisher Information Matrix Constructing')
                    if n == 0:
                        fisher = {}
                        for name, param in _use_model.named_parameters():
                            if not name.startswith(('fc', 'bn')):
                                fisher[name] = torch.zeros(param.size()).to(d)
                    fisher = FisherForEWC(_use_model, data, fisher, self.args)
                    
                    # Omega For MAS
                    print('Omega Constructing')
                    if n == 0:
                        omega = {}
                        for name, param in _use_model.named_parameters():
                            if not name.startswith(('fc', 'bn')):
                                omega[name] = torch.zeros(param.size()).to(d)
                    omega = OmegaForMAS(_use_model, data, omega, self.args)
                    print('Done')
                    
                    
                    synthesized += LoadSynthesized(site, d, self.args)
            
            
        
        return _get_weight, local_trainer            

























