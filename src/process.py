import pandas as pd
import numpy as np
import os
import logging
import copy
import math
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import CEWeightCalculator, useDataset, getauc, \
    knowledge_distillation_loss, seedset, Components

pd.set_option('display.max_columns', None)



class LwF(Components):
    def __init__(self, train_loader, val_loader, model, savepath, args, weight=None): 
        super().__init__()
        self.get_joint_components(train_loader, val_loader, model, savepath, args, weight)
    

    def train(self, local_trainer, curr_site, auc_threshold):
        if not os.path.exists(self.savepath): os.mkdir(self.savepath)
         
        d = self.device
        bestauc = 0.0
        patience = 0
        self._init_model = copy.deepcopy(self.model)
        self._init_model.eval()
        self._init_weight = self._init_model.state_dict()
        # self.sub_epoch = self.args.sub_epoch
        
        # self.reg_model = copy.deepcopy(self.model)
        # self.reg_optimizer = optim.Adam(self.reg_model.parameters(), lr=self.args.lr)
        
        
        logging.debug('LwF Learning Start')
        self._lambda = 1
        while bestauc <= auc_threshold:
            bestauc = 0.0
            
            # logging.debug(f'Sub Epoch: {self.sub_epoch}')
            logging.debug(f'Reg Lambda: {self._lambda}')
            self.model.load_state_dict(self._init_weight)
            for e in range(self.args.epoch):
                self.model.train()
                target_loss_sum = 0
                dist_loss_sum = 0
                
                skipcnt = 0
                for age, sex, ecg, flag in tqdm(self.train_loader):
                    
                    if ecg.shape[0] == 1:
                        skipcnt += 1
                        continue
                    
                    age, sex, ecg, flag = age.to(d), sex.to(d), ecg.to(d), flag.to(d)
                    
                    output = self.model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
                    curr_loss = self.criterion(output, flag.type(torch.long))
                    target_loss_sum += curr_loss.item()
                    
                    # self.optimizer.zero_grad()
                    # curr_loss.backward()
                    # self.optimizer.step()
                    
                    # self.reg_model.load_state_dict(self.model.state_dict())
                    # for _epoch in range(self.sub_epoch):
                        # self.reg_model.train()
                        
                    init_output = self._init_model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
                    dist_loss = self.kdloss.get(output, flag.type(torch.long), init_output)
                    # self.reg_optimizer.zero_grad()
                    # dist_loss.backward()
                    # self.reg_optimizer.step()
                    
                    loss = curr_loss + self._lambda * dist_loss
                    loss.backward()
                    self.optimizer.step()
                    
                    dist_loss_sum += dist_loss.item()
                    # self.model.load_state_dict(self.reg_model.state_dict())
                    

            
                if (e + 1) % self.args.valtime == 0: # 얼리스타핑 사용할땐 주석처리
                    div = len(self.train_loader) - skipcnt
                    val_loss, target_loss, dist_loss, roc_auc, pr_auc = self.evaluation(self.val_loader, self._init_model, None)
                    logging.debug('Epoch [{}/{}], Train Loss: {:.4f}, Target Loss: {:.4f}, Dist Loss: {:.4f}'.format(
                                e+1, self.args.epoch, (target_loss_sum + dist_loss_sum) / div, target_loss_sum / div, dist_loss_sum / div))
                    logging.debug('Epoch [{}/{}], Val Loss: {:.4f}, Target Loss: {:.4f}, Dist Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}'.format(
                                e+1, self.args.epoch, val_loss, target_loss, dist_loss, roc_auc, pr_auc))
                        
                    if roc_auc > bestauc + 0.001:
                        logging.debug(f'Saved Best Model...')
                        bestauc = roc_auc
                        torch.save(self.model.state_dict(), self.savepath + f'best_supervised_model_LwF.pt')
                        
                        local_trainer[curr_site] = {
                            name: param for name, param in self.model.named_parameters() \
                                if name.startswith(('fc', 'bn'))}
                        
                        torch.save(local_trainer, self.savepath + f'best_supervised_local_model_LwF.pt')
                        patience = 0
                    else:
                        patience += 1
    
                
                if self.args.es and patience > self.args.patience:
                    logging.debug(f'Early Stopping Activated')
                    break
            
            # if bestauc <= auc_threshold: self.sub_epoch -= 1
            if bestauc <= auc_threshold: self._lambda *= self.args.alpha
            
            
        load_model = torch.load(self.savepath + f'best_supervised_model_LwF.pt', map_location=self.device)
        load_local = torch.load(self.savepath + f'best_supervised_local_model_LwF.pt', map_location=self.device)
            
        # return load_model, load_local, self.sub_epoch
        return load_model, load_local, self._lambda
        

    def evaluation(self, dataloader, init_model, _dump):
        d = self.device
        self.model.eval()
        init_model.eval()
        vloss, flags = [], []
        
        loss_sum = 0
        target_loss_sum = 0
        dist_loss_sum = 0
        
        skipcnt = 0
        for age, sex, ecg, flag in dataloader:
            
            if ecg.shape[0] == 1:
                skipcnt += 1
                continue
            
            age, sex, ecg, flag = age.to(d), sex.to(d), ecg.to(d), flag.to(d)
            
            output = self.model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
            curr_loss = self.criterion(output, flag.type(torch.long))
            init_output = init_model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
            dist_loss = self.kdloss.get(output, flag.type(torch.long), init_output)
            
            loss = curr_loss + dist_loss
            loss_sum += loss.item()
            target_loss_sum += curr_loss.item()
            dist_loss_sum += dist_loss.item()
            
            vloss.append(output[:, 1].view(-1).cpu().detach().numpy())
            flags.append(flag.view(-1).cpu().detach().numpy().astype(int))
            
        
        vloss = np.concatenate(vloss)
        flags = np.concatenate(flags)

        fpr, tpr, _ = roc_curve(flags, vloss)
        precision, recall, _ = precision_recall_curve(flags, vloss)
        roc_auc = auc(fpr, tpr)
        pr_auc =  auc(recall, precision)
        
        div = len(dataloader) - skipcnt
        
        return loss_sum / div, target_loss_sum / div, dist_loss_sum / div, roc_auc, pr_auc





class EWC(Components):
    def __init__(self, train_loader, val_loader, model, savepath, args, weight=None): 
        super().__init__()
        self.get_joint_components(train_loader, val_loader, model, savepath, args, weight)
    
    
    def train(self, local_trainer, curr_site, fisher, auc_threshold):
        if not os.path.exists(self.savepath): os.mkdir(self.savepath)
        
        d = self.device
        bestauc = 0.0
        patience = 0
        self._init_model = copy.deepcopy(self.model)
        self._init_model.eval()
        self._init_weight = self._init_model.state_dict()
        # self.sub_epoch = self.args.sub_epoch
        
        # self.reg_model = copy.deepcopy(self.model)
        # self.reg_optimizer = optim.Adam(self.reg_model.parameters(), lr=self.args.lr)
        
        
        logging.debug('EWC Learning Start')
        self._lambda = 1
        while bestauc <= auc_threshold:
            bestauc = 0.0
            
            # logging.debug(f'Sub Epoch: {self.sub_epoch}')
            logging.debug(f'Reg Lambda: {self._lambda}')
            self.model.load_state_dict(self._init_weight)
            for e in range(self.args.epoch):
                self.model.train()
                all_loss_sum = 0
                target_loss_sum = 0
                
                skipcnt = 0
                for age, sex, ecg, flag in tqdm(self.train_loader):
                    
                    if ecg.shape[0] == 1:
                        skipcnt += 1
                        continue
                    
                    age, sex, ecg, flag = age.to(d), sex.to(d), ecg.to(d), flag.to(d)
                    
                    output = self.model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
                    curr_loss = self.criterion(output, flag.type(torch.long))
                    target_loss_sum += curr_loss.item()
                    # self.optimizer.zero_grad()
                    # curr_loss.backward()
                    # self.optimizer.step()
                    
                    # self.reg_model.load_state_dict(self.model.state_dict())
                    # for _epoch in range(self.sub_epoch):
                    #     self.reg_model.train()
                        
                    for name, param in self.model.named_parameters():
                        if name not in list(fisher.keys()): continue
                        diff = param - self._init_model.state_dict()[name]
                        diff_square = diff**2
                        _loss = fisher[name] * diff_square 
                        curr_loss += self._lambda * _loss.sum()
                            
                    # self.reg_optimizer.zero_grad()
                    # ewc_loss.backward()
                    # self.reg_optimizer.step()
                    
                    curr_loss.backward()
                    self.optimizer.step()
                    
                    all_loss_sum += curr_loss.item()
                    # self.model.load_state_dict(self.reg_model.state_dict())
                    
            
                if (e + 1) % self.args.valtime == 0: # 얼리스타핑 사용할땐 주석처리
                    div = len(self.train_loader) - skipcnt
                    val_loss, val_target, val_mas, roc_auc, pr_auc = self.evaluation(self.val_loader, self._init_model, fisher)
                    logging.debug('Epoch [{}/{}], Train Loss: {:.4f}, Target Loss: {:.4f}, EWC Loss: {:.4f}'.format(
                            e+1, self.args.epoch, all_loss_sum / div, target_loss_sum / div, (all_loss_sum - target_loss_sum) / div))
                    logging.debug('Epoch [{}/{}], Val Loss  : {:.4f}, Target Loss: {:.4f}, EWC Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}'.format(
                            e+1, self.args.epoch, val_loss, val_target, val_mas, roc_auc, pr_auc))

                    if roc_auc > bestauc + 0.001:
                        logging.debug(f'Saved Best Model...')
                        bestauc = roc_auc
                        torch.save(self.model.state_dict(), self.savepath + 'best_supervised_model_EWC.pt')
                        
                        local_trainer[curr_site] = {
                            name: param for name, param in self.model.named_parameters() \
                                if name.startswith(('fc', 'bn'))}
                        
                        torch.save(local_trainer, self.savepath + 'best_supervised_local_model_EWC.pt')
                        patience = 0
                    else:
                        patience += 1
    
                if self.args.es and patience > self.args.patience:
                    logging.debug(f'Early Stopping Activated')
                    break
            
            # if bestauc <= auc_threshold: self.sub_epoch -= 1
            if bestauc <= auc_threshold: self._lambda *= self.args.alpha
            
        load_model = torch.load(self.savepath + 'best_supervised_model_EWC.pt', map_location=self.device)
        load_local = torch.load(self.savepath + 'best_supervised_local_model_EWC.pt', map_location=self.device)
            
        # return load_model, load_local, self.sub_epoch
        return load_model, load_local, self._lambda


    def evaluation(self, dataloader, init_model, fisher):
        d = self.device
        self.model.eval()
        vloss, flags = [], []
        
        loss_sum = 0
        target_loss_sum = 0
        ewc_loss_sum = 0
        
        skipcnt = 0
        for age, sex, ecg, flag in dataloader:
            
            if ecg.shape[0] == 1:
                skipcnt += 1
                continue
            
            age, sex, ecg, flag = age.to(d), sex.to(d), ecg.to(d), flag.to(d)
            
            output = self.model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
            curr_loss = self.criterion(output, flag.type(torch.long))
            target_loss_sum += curr_loss.item()
                
            for name, param in self.model.named_parameters():
                if name not in list(fisher.keys()): continue
                diff = param - init_model.state_dict()[name]
                diff_square = diff**2
                _loss = fisher[name] * diff_square # F_i * (theta_i  - theta'_i) ** 2
                ewc_loss = _loss.sum()
                curr_loss += ewc_loss
                ewc_loss_sum += ewc_loss.item()
            loss_sum += curr_loss.item()
            
            vloss.append(output[:, 1].view(-1).cpu().detach().numpy())
            flags.append(flag.view(-1).cpu().detach().numpy().astype(int))
        
        vloss = np.concatenate(vloss)
        flags = np.concatenate(flags)

        fpr, tpr, _ = roc_curve(flags, vloss)
        precision, recall, _ = precision_recall_curve(flags, vloss)
        roc_auc = auc(fpr, tpr)
        pr_auc =  auc(recall, precision)
        
        div = len(dataloader) - skipcnt
        
        return loss_sum / div, target_loss_sum / div, ewc_loss_sum / div, roc_auc, pr_auc



class MAS(Components):
    def __init__(self, train_loader, val_loader, model, savepath, args, weight=None): 
        super().__init__()
        self.get_joint_components(train_loader, val_loader, model, savepath, args, weight)
    

    def train(self, local_trainer, curr_site, omega, auc_threshold):
        if not os.path.exists(self.savepath): os.mkdir(self.savepath)
        
        d = self.device
        bestauc = 0.0
        patience = 0
        self._init_model = copy.deepcopy(self.model)
        self._init_model.eval()
        self._init_weight = self._init_model.state_dict()
        
        
        logging.debug('MAS Learning Start')
        self._lambda = 1
        while bestauc <= auc_threshold:
            bestauc = 0.0
            
            # logging.debug(f'Sub Epoch: {self.sub_epoch}')
            logging.debug(f'Reg Lambda: {self._lambda}')
            self.model.load_state_dict(self._init_weight)
            for e in range(self.args.epoch):
                self.model.train()
                all_loss_sum = 0
                target_loss_sum = 0
                
                skipcnt = 0
                for age, sex, ecg, flag in tqdm(self.train_loader):
                    
                    if ecg.shape[0] == 1:
                        skipcnt += 1
                        continue
                    
                    age, sex, ecg, flag = age.to(d), sex.to(d), ecg.to(d), flag.to(d)
                    
                    output = self.model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
                    curr_loss = self.criterion(output, flag.type(torch.long))
                    target_loss_sum += curr_loss.item()
                    
                    # mas_loss = 0
                    for name, param in self.model.named_parameters():
                        if name not in list(omega.keys()): continue
                        diff = param - self._init_weight[name]
                        diff_square = diff**2
                        # mas_loss += torch.sum(torch.mul(diff_square, omega[name])).item()
                        curr_loss += self._lambda * torch.sum(torch.mul(diff_square, omega[name]))
                    
                    curr_loss.backward()
                    self.optimizer.step()
                    
                    all_loss_sum += curr_loss.item()
            
            
                if (e + 1) % self.args.valtime == 0: # 얼리스타핑 사용할땐 주석처리
                    div = len(self.train_loader) - skipcnt
                    val_loss, val_target, val_mas, roc_auc, pr_auc = self.evaluation(self.val_loader, self._init_model, omega, self._lambda)
                    logging.debug('Epoch [{}/{}], Train Loss: {:.4f}, Target Loss: {:.4f}, MAS Loss: {:.4f}'.format(
                            e+1, self.args.epoch, all_loss_sum / div, target_loss_sum / div, (all_loss_sum - target_loss_sum) / div))
                    logging.debug('Epoch [{}/{}], Val Loss  : {:.4f}, Target Loss: {:.4f}, MAS Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}'.format(
                            e+1, self.args.epoch, val_loss, val_target, val_mas, roc_auc, pr_auc))

                    if roc_auc > bestauc + 0.001:
                        logging.debug(f'Saved Best Model...')
                        bestauc = roc_auc
                        torch.save(self.model.state_dict(), self.savepath + 'best_supervised_model_MAS.pt')
                        
                        local_trainer[curr_site] = {
                            name: param for name, param in self.model.named_parameters() \
                                if name.startswith(('fc', 'bn'))}
                        
                        torch.save(local_trainer, self.savepath + 'best_supervised_local_model_MAS.pt')
                        patience = 0
                    else:
                        patience += 1
    
                if self.args.es and patience > self.args.patience:
                    logging.debug(f'Early Stopping Activated')
                    break
                
            if bestauc <= auc_threshold: self._lambda *= self.args.alpha
            
        load_model = torch.load(self.savepath + 'best_supervised_model_MAS.pt', map_location=self.device)
        load_local = torch.load(self.savepath + 'best_supervised_local_model_MAS.pt', map_location=self.device)
            
        return load_model, load_local, self._lambda


    def evaluation(self, dataloader, init_model, omega, _lambda):
        d = self.device
        self.model.eval()
        vloss, flags = [], []
        
        loss_sum = 0
        target_loss_sum = 0
        mas_loss_sum = 0
        
        skipcnt = 0
        for age, sex, ecg, flag in dataloader:
            
            if ecg.shape[0] == 1:
                skipcnt += 1
                continue
            
            age, sex, ecg, flag = age.to(d), sex.to(d), ecg.to(d), flag.to(d)
            
            output = self.model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
            curr_loss = self.criterion(output, flag.type(torch.long))
            target_loss_sum += curr_loss.item()
                
            mas_loss = 0
            for name, param in self.model.named_parameters():
                if name not in list(omega.keys()): continue
                diff = param - init_model.state_dict()[name]
                diff_square = diff**2
                mas_loss += torch.sum(torch.mul(diff_square, omega[name]) * _lambda)
                
                curr_loss += mas_loss
                mas_loss_sum += mas_loss.item()
            loss_sum += curr_loss.item()
            
            vloss.append(output[:, 1].view(-1).cpu().detach().numpy())
            flags.append(flag.view(-1).cpu().detach().numpy().astype(int))
        
        vloss = np.concatenate(vloss)
        flags = np.concatenate(flags)

        fpr, tpr, _ = roc_curve(flags, vloss)
        precision, recall, _ = precision_recall_curve(flags, vloss)
        roc_auc = auc(fpr, tpr)
        pr_auc =  auc(recall, precision)
        
        div = len(dataloader) - skipcnt
        
        return loss_sum / div, target_loss_sum / div, mas_loss_sum / div, roc_auc, pr_auc




