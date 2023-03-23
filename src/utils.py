from scipy.signal import butter, lfilter
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from functools import reduce
import pickle
from tqdm import tqdm
import random
import copy
import matplotlib.pyplot as plt
import neurokit2 as nk

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.p2pmodel import Pulse2pulseGenerator


def aucfloat(f):
    if float(f) < 0.5:
        return 1 - float(f)
    else:
        return float(f)
    
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter_and_normalization(data, lowcut, highcut, fs, order=5):
    yall = []
    for dat in data:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        bp = lfilter(b, a, dat)
        # Normalize between -1 and 1
        y = 2*(bp - np.min(bp)) / (np.max(bp)-np.min(bp))
        y = y - 1
        yall.append(y)
    
    return np.array(yall)


def loadpickle(file):
    with open(file, 'rb') as f:
        openfile = pickle.load(f)
    openfile = [i for i in openfile if i != 0]
    return openfile


def sample_noise(size):
    z = torch.FloatTensor(size, 12, 5000)
    z.data.normal_()  # generating latent space based on normal distribution
    return z


def LoadData(seed):
    flist = ['shaoxing.pickle', 'ptb.pickle', 'georgia.pickle', 'cpsc.pickle']
    
    db = {d.split('.')[0]: loadpickle(f'data/{d}') for d in flist}

    train_site, local_site, global_site = {}, {}, {}
    
    for site, data in db.items():
        _train, global_site[site] = train_test_split(data, test_size=1/10, random_state=seed)
        train_site[site], local_site[site] = train_test_split(_train, test_size=1/9, random_state=seed)

    return train_site, local_site, global_site


def LoadSingleData(seed, file):
    with open(f'data/{file}_single.pickle', 'rb') as f:
        lfile = pickle.load(f)
           
    train_site, local_site, global_site = {}, {}, {}
    
    for site, data in lfile.items():
        _train, global_site[site] = train_test_split(data, test_size=1/10, random_state=seed)
        train_site[site], local_site[site] = train_test_split(_train, test_size=1/9, random_state=seed)

    return train_site, local_site, global_site


def LoadSynthesized(site, device, args):
    sample_size = 2000
    if args.domain == 'multiple':
        with open(f'data/{site}.pickle', 'rb') as f:
            ttmp = pickle.load(f)
    else:
        with open(f'data/{args.file}_single.pickle', 'rb') as f:
            ttmp = pickle.load(f)
            ttmp = ttmp[site]
        
    openfile = [i for i in ttmp if i != 0]
    dat = pd.DataFrame(openfile, columns=['age', 'sex', 'ecg', 'flag'])
    flag0, flag1 = dat[dat['flag'] == 0], dat[dat['flag'] == 1]
    sample0 = flag0.sample(n=int(sample_size * flag0.shape[0] / dat.shape[0]), random_state=567)
    sample1 = flag1.sample(n=int(sample_size * flag1.shape[0] / dat.shape[0]), random_state=567)

    lead = 12
    d = 50
    generator = Pulse2pulseGenerator(model_size=d, num_channels=lead).to(device)
    if args.domain == 'multiple':
        ganpath = f'results/{args.domain}/pulse2pulse/saved/pulse2pulse_{site}_0_64_0.0001_50.tar'
    else:
        ganpath = f'results/{args.domain}/pulse2pulse/{args.file}/saved/pulse2pulse_{site}_0_64_0.0001_50.tar'
    
    saved = torch.load(ganpath, map_location=device)
    generator.load_state_dict(saved['generator'])
    tmp = sample_noise(sample0.shape[0]).to(device)
    signal0 = generator(tmp)

    if args.domain == 'multiple':
        ganpath = f'results/{args.domain}/pulse2pulse/saved/pulse2pulse_{site}_1_64_0.0001_50.tar'
    else:
        ganpath = f'results/{args.domain}/pulse2pulse/{args.file}/saved/pulse2pulse_{site}_1_64_0.0001_50.tar'
    
    saved = torch.load(ganpath, map_location=device)
    generator.load_state_dict(saved['generator'])
    tmp = sample_noise(sample1.shape[0]).to(device)
    signal1 = generator(tmp)

    finaldat = [
        [sample0.iloc[n]['age'], sample0.iloc[n]['sex'], s.cpu().detach().numpy(), 0] for n, s in enumerate(signal0)
    ] + [
        [sample1.iloc[n]['age'], sample1.iloc[n]['sex'], s.cpu().detach().numpy(), 1] for n, s in enumerate(signal1)
    ]

    return finaldat


class useDataset(Dataset):
    def __init__(self, dat):
        self.dat = dat
        self.age = torch.Tensor(np.array([i[0] for i in dat])) / 40 - 1
        self.sex = torch.Tensor(np.array([i[1] for i in dat]))
        self.ecg = torch.Tensor(np.array([i[2] for i in dat]))
        self.flag = torch.Tensor(np.array([i[3] for i in dat]))
        
        
    def __len__(self):
        return len(self.dat)
    
    
    def __getitem__(self, idx):
        # Return Age, Sex, Signal
        return self.age[idx], self.sex[idx], self.ecg[idx], self.flag[idx]



class Components:
    def __init__(self):
        pass
    
    
    def get_joint_components(self, train_loader, val_loader, model, savepath, args, weight=None): 
        self.args = args
        seedset(args.seed)
        if args.device != 'cpu':
            self.device = f'cuda:{args.device}'
        else:
            self.device = args.device
         
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.savepath = savepath
        
        if weight == None:
            self.weight = CEWeightCalculator(train_loader)
        else:
            self.weight = weight
        
        self.kdloss = knowledge_distillation_loss()
        self.criterion = nn.CrossEntropyLoss(weight=self.weight).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        
        
    def get_separated_components(self, train_site_data, local_site_data, model, savepath, args):
        self.args = args
        seedset(args.seed)
        if args.device != 'cpu':
            self.device = f'cuda:{args.device}'
        else:
            self.device = args.device
            
        self.model = model
        self.train_site_data = train_site_data
        self.local_site_data = local_site_data
        self.savepath = savepath
        
        all_len = sum([len(v) for k, v in train_site_data.items()])
        self.coefficients = {
            k: len(v) / all_len for k, v in train_site_data.items()
        }
        
        self.kdloss = knowledge_distillation_loss()
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    
    def evaluation(self, model, dataloader):
        d = self.device
        model.eval()
        vloss, flags = [], []
        
        loss_sum = 0
        skipcnt = 0
        for age, sex, ecg, flag in dataloader:
            
            if ecg.shape[0] == 1:
                skipcnt += 1
                continue
            
            age, sex, ecg, flag = age.to(d), sex.to(d), ecg.to(d), flag.to(d)
            
            output = model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
            loss = self.criterion(output, flag.type(torch.long))
            loss_sum += loss.item()
            vloss.append(output[:, 1].view(-1).cpu().detach().numpy())
            flags.append(flag.view(-1).cpu().detach().numpy().astype(int))
        
        vloss = np.concatenate(vloss)
        flags = np.concatenate(flags)

        fpr, tpr, _ = roc_curve(flags, vloss)
        precision, recall, _ = precision_recall_curve(flags, vloss)
        roc_auc = auc(fpr, tpr)
        pr_auc =  auc(recall, precision)
        
        return loss_sum / (len(dataloader) - skipcnt), roc_auc, pr_auc  




def multiply(arr):
    return reduce(lambda x, y: x * y, arr)


def CEWeightCalculator(dataloader):
    size = 0
    _w = 0
    print('Weight Calculating...')
    
    for age, sex, ecg, flag in tqdm(dataloader):
        size += multiply(flag.cpu().numpy().shape)
        _w += torch.sum(flag).item()

    weight = [(size - _w) / size, _w / size]
    return torch.Tensor(np.array(weight))


def sitesum(site_data):
    all = []
    for k, v in site_data.items():
        all += v
    
    return all


def getauc(flags, vloss):
    fpr, tpr, thres = roc_curve(flags, vloss)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thres = thres[ix]
    y_prob_pred = (vloss >= best_thres).astype(bool)
    precision, recall, _ = precision_recall_curve(flags, vloss)
    cf = confusion_matrix(flags, y_prob_pred)
    
    roc_auc = auc(fpr, tpr)
    pr_auc =  auc(recall, precision)
    
    return best_thres, cf, roc_auc, pr_auc


class knowledge_distillation_loss:
    def __init__(self, alpha=0.1, T=10):
        self.alpha = alpha
        self.T = T
    
    
    def make_softmax(self, arr):
        return torch.Tensor([[1-i, i] for i in torch.sigmoid(arr)])
        
    
    def get(self, logits, labels, teacher_logits):
        student_loss = F.cross_entropy(input=logits, target=labels)
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(logits/self.T, dim=1), F.softmax(teacher_logits/self.T, dim=1)) * (self.T * self.T)
        total_loss =  self.alpha*student_loss + (1-self.alpha)*distillation_loss

        return total_loss
    

def seedset(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    


def OmegaForMAS(model, data, omega, args):
    d = f'cuda:{args.device}' if args.device != 'cpu' else args.device
    model.eval()
    _dl = DataLoader(useDataset(data), batch_size=args.bs, shuffle=True)
    
    new_omega = {}
    for name, param in model.named_parameters():
        if not name.startswith(('fc', 'bn')):
            new_omega[name] = torch.zeros(param.size()).to(d)
    
    for batch_index, (age, sex, ecg, flag) in tqdm(enumerate(_dl)):
        model.zero_grad()
        age, sex, ecg, flag = age.to(d), sex.to(d), ecg.to(d), flag.to(d)
        outputs = model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
        l2_norm = torch.norm(outputs, 2, dim = 1)
        squared_l2_norm = l2_norm**2
        sum_norm = torch.sum(squared_l2_norm)
        sum_norm.backward()
        
        for name, param in model.named_parameters():
            if not name.startswith(('fc', 'bn')):
                grad_data_copy = param.grad.data.clone().abs()

                current_size = (batch_index+1)*args.bs
                step_size = 1/float(current_size)
                
                new_omega[name] = new_omega[name] + step_size*(grad_data_copy - args.bs*new_omega[name])
    
    for name in omega.keys():
        if not name.startswith(('fc', 'bn')):
            omega[name] += new_omega[name]
        
    return omega


def FisherForEWC(model, data, fisher, args):
    d = f'cuda:{args.device}' if args.device != 'cpu' else args.device
    model.eval()
    _dl = DataLoader(useDataset(data), batch_size=args.bs, shuffle=True)
    
    precision_matrices = {}
    model.eval()
    for age, sex, ecg, flag in tqdm(_dl):
        age, sex, ecg, flag = age.to(d), sex.to(d), ecg.to(d), flag.to(d)
        model.zero_grad()
        flag = flag.view(-1).type(torch.LongTensor).to(d)
        output = model(ecg, torch.cat((age.view(-1, 1), sex.view(-1, 1)), dim=1))
        # label = output.max(1)[1].view(-1)
        
        loss = F.nll_loss(F.log_softmax(output, dim=1), flag)
        loss.backward()

        for n, p in model.named_parameters():
            if not n.startswith(('fc', 'bn')):
                precision_matrices[n] = p.grad.data.pow(2) / len(_dl) # grad = first order derivatives; point (b) - EWC paper

    # precision_matrices = {n: p for n, p in precision_matrices.items() if not n.startswith(('fc', 'bn'))}
    # _mean = torch.mean(torch.cat([p.view(-1) for n, p in precision_matrices.items()]))
    # precision_matrices = {n: p / _mean for n, p in precision_matrices.items()}
    
    # precision_matrices = {n: p for n, p in precision_matrices.items() if not n.startswith(('fc', 'bn'))}
    # _mean = torch.mean(torch.cat([p.view(-1) for n, p in precision_matrices.items()])) * 2000
    # precision_matrices = {n: p / _mean for n, p in precision_matrices.items()}
    
    
    for name in precision_matrices.keys():
        precision_matrices[name] += fisher[name]
    
    return precision_matrices


def saveplot_12lead(savepth, bs, sample, epoch, lead=1, model='wavegan'):
    os.makedirs(savepth, exist_ok=True)
    s = [sample[0], sample[1], sample[2], sample[3]]
    
    plt.figure(figsize=(60,20))
    cnt = 0
    for i in range(12):
        for j in range(4):
            plt.subplot(12, 4, cnt+1)
            plt.plot(s[j][i])
            cnt += 1
    plt.tight_layout()
    plt.savefig(os.path.join(savepth, f'fixnoise_epoch_{epoch}.png'))
    plt.close()


def ecg_feature_matrix(wf):
    try:
        _, rpeaks = nk.ecg_peaks(wf[1], sampling_rate=500)
        signal_cwt, waves_cwt = nk.ecg_delineate(wf[1], rpeaks, sampling_rate=500, method="dwt")
        
        waves_cwt['ECG_R_Peaks'] = rpeaks['ECG_R_Peaks']
        waves_cwt['ECG_postR_Peaks'] = list(rpeaks['ECG_R_Peaks'][1:]) + [0]
        waves_cwt['ECG_postP_Peaks'] = list(waves_cwt['ECG_P_Peaks'][1:])+ [0]
        waves_cwt['ECG_postT_Peaks'] = list(waves_cwt['ECG_T_Peaks'][1:]) + [0]
        waves_cwt['ECG_postP_Onsets'] = list(waves_cwt['ECG_P_Onsets'][1:]) + [0]

        waves_cwt = pd.DataFrame(waves_cwt)
        waves_cwt = waves_cwt.iloc[:-2]

        waves_cwt['P_wave_duration'] = (waves_cwt['ECG_P_Offsets'] - waves_cwt['ECG_P_Onsets']) / 500
        waves_cwt['PR_interval'] = (waves_cwt['ECG_R_Onsets'] - waves_cwt['ECG_P_Onsets']) / 500
        waves_cwt['PP_interval'] = (waves_cwt['ECG_postP_Peaks'] - waves_cwt['ECG_P_Peaks']) / 500
        waves_cwt['PR_segment'] = (waves_cwt['ECG_R_Onsets'] - waves_cwt['ECG_P_Offsets']) / 500
        waves_cwt['QRS_duration'] = (waves_cwt['ECG_R_Offsets'] - waves_cwt['ECG_R_Onsets']) / 500
        waves_cwt['QT_duration'] = (waves_cwt['ECG_T_Offsets'] - waves_cwt['ECG_R_Onsets']) / 500
        waves_cwt['RR_interval'] = (waves_cwt['ECG_postR_Peaks'] - waves_cwt['ECG_R_Peaks']) / 500
        waves_cwt['ST_segment'] = (waves_cwt['ECG_T_Onsets'] - waves_cwt['ECG_R_Offsets']) / 500
        waves_cwt['ST_T_segment'] = (waves_cwt['ECG_T_Offsets'] - waves_cwt['ECG_R_Offsets']) / 500
        waves_cwt['TP_interval'] = (waves_cwt['ECG_postP_Onsets'] - waves_cwt['ECG_T_Offsets']) / 500
        
        ecg_features = np.array(waves_cwt[ecg_feature_cols].mean())
        
        return ecg_features
    
    except:
        return [0]