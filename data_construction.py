import pandas as pd
import numpy as np
import os
import ray
import wfdb
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils import butter_bandpass_filter_and_normalization, loadpickle


if not os.path.exists('data'): os.mkdir('data')



# PTB-XL (physionet)
print('PTB-XL Contructing...')
if not os.path.exists('data/ptb.pickle'):
    fpath = '../ecgs/ptb-xl/1.0.2/'
    train_files = pd.read_csv(fpath + "ptbxl_database.csv")
    ecgs = {}

    ray.init(num_cpus=112)

    @ray.remote
    def denoising(t):
        record = wfdb.rdrecord(fpath + train_files.loc[t]['filename_hr'])
        rec_dict = record.__dict__
        lead1and2 = np.transpose(rec_dict['p_signal'])
        bp = butter_bandpass_filter_and_normalization(lead1and2, 0.5, 40, 500, 5)
            
        tmp = eval(train_files.loc[t]['scp_codes'])
        flag = 0 if 'NORM' in tmp.keys() else 1
        
        if np.sum(np.isnan(bp)) != 0:
            return 0
        elif float(train_files.loc[t]['age']) < 18 or float(train_files.loc[t]['age']) > 100:
            return 0
        else:
            return [train_files.loc[t]['age'], train_files.loc[t]['sex'], bp, flag]

    train_ecg = [denoising.remote(i) for i in tqdm(train_files.index)]
    train_ecg = ray.get(train_ecg)

    ray.shutdown()
    with open('data/ptb.pickle', 'wb') as f:
        pickle.dump(train_ecg, f)
print('Done')
    

    
# CPSC2018 (kaggle)
print('CPSC2018 Constructing...')
if not os.path.exists('data/cpsc.pickle'):
    fpath = '../ecgs/cpsc2018/Training_WFDB/'
    codes = pd.read_csv("../ecgs/cpsc2018/REFERENCE.csv")
    flist = list(set([i.split('.')[0] for i in os.listdir(fpath)]))
    flist.sort()

    ray.init(num_cpus=112)

    @ray.remote
    def fsave(fname):
        record = wfdb.rdrecord(fpath + fname).__dict__
        cond1 = record['sig_len'] >= 5000
        cond2 = record['fs'] == 500
        cond3 = record['n_sig'] == 12

        sexdict = {'Male': 0, 'Female': 1}

        if cond1 and cond2 and cond3:
            try:
                _sig = record['p_signal'][:5000, :]
                sig = np.transpose(_sig)
                bp = butter_bandpass_filter_and_normalization(sig, 0.5, 40, 500, 5)
                age = int(record['comments'][0].split(': ')[-1])
                sex = sexdict[record['comments'][1].split(': ')[-1]]
                dx = codes[codes['Recording'] == fname]['First_label'].iloc[0]
                dx = 0 if dx == 1 else 1
                
                
                if np.sum(np.isnan(bp)) != 0:
                    return 0
                elif float(age) < 18 or float(age) > 100:
                    return 0
                else:
                    return [age, sex, bp, dx]
            
            except:
                return 0
        
        else:
            return 0

    train_ecg = [fsave.remote(fname) for fname in tqdm(flist)]
    train_ecg = ray.get(train_ecg)    
    ray.shutdown()

    with open('data/cpsc.pickle', 'wb') as f:
        pickle.dump(train_ecg, f)
print('Done')



# Shaoxing and Ningbo (physionet)
print('Shaoxing and Ningbo Constructing...')
if not os.path.exists('data/shaoxing.pickle'):
    codes = pd.read_csv("../ecgs/ecg-arrhythmia/1.0.0/ConditionNames_SNOMED-CT.csv")
    file = open("../ecgs/ecg-arrhythmia/1.0.0/SHA256SUMS.txt", "r")
    flist = []
    while True:
        line = file.readline()
        if not line:
            break
        flist.append(line.strip())

    file.close()

    fpath = '../ecgs/ecg-arrhythmia/1.0.0/'

    flist = list(set([i.split(' ')[-1].split('.')[0] for i in flist if 'JS' in i]))
    flist.sort()

    ray.init(num_cpus=112)

    @ray.remote
    def fsave(fname):
        
        try:
            record = wfdb.rdrecord(fpath + fname).__dict__
            cond1 = record['sig_len'] >= 5000
            cond2 = record['fs'] == 500
            cond3 = record['n_sig'] == 12

            sexdict = {'Male': 0, 'Female': 1}

            if cond1 and cond2 and cond3:
                _sig = record['p_signal'][:5000, :]
                sig = np.transpose(_sig)
                bp = butter_bandpass_filter_and_normalization(sig, 0.5, 40, 500, 5)
                age = int(record['comments'][0].split(': ')[-1])
                sex = sexdict[record['comments'][1].split(': ')[-1]]
                
                # SR,Sinus Rhythm,426783006
                dx = 0 if '426783006' in record['comments'][2].split(': ')[-1] else 1
                
                if np.sum(np.isnan(bp)) != 0:
                    return 0
                elif float(age) < 18 or float(age) > 1000:
                    return 0
                else:
                    return [age, sex, bp, dx]
            
            else:
                return 0
            
        except:
            return 0
        

    train_ecg = [fsave.remote(fname) for fname in tqdm(flist)]
    train_ecg = ray.get(train_ecg)    
    ray.shutdown()
    train_ecg = [i for i in train_ecg if i != 0]

    with open('data/shaoxing.pickle', 'wb') as f:
        pickle.dump(train_ecg, f)
print('Done')



# Georgia (physionet)
print('Georgia Constructing...')
if not os.path.exists('data/georgia.pickle'):
    fpath = '../ecgs/georgia/WFDB_Ga/'
    flist = list(set([i.split('.')[0] for i in os.listdir(fpath)]))
    flist.sort()

    ray.init(num_cpus=112)
    train_ecg = [fsave.remote(fname) for fname in tqdm(flist)]
    train_ecg = ray.get(train_ecg)    
    ray.shutdown()
    train_ecg = [i for i in train_ecg if i != 0]

    with open('data/georgia.pickle', 'wb') as f:
        pickle.dump(train_ecg, f)
print('Done')



# Single data processing
flist = ['ptb']
for f in flist:
    print(f'Single {f} Constructing...')
    db = loadpickle(f'data/{f}.pickle')
    db = pd.DataFrame(db, columns=['age', 'sex', 'ecg', 'flag'])
    group = {}
    agegroup = [[18, 60], [60, 100]]
    sexgroup = [0, 1]

    for a in agegroup:
        for s in sexgroup:
            group[f'{a[0]}_{a[1]}_{s}'] = db[(db['age'].between(a[0], a[1], 'left')) & (db['sex'] == s)]

    final_group = {k: [] for k in group.keys()}
    for g, d in group.items():
        
        shuffle = d.sample(frac=1)
        idx_group = np.array_split(shuffle.index, 10)
        final_group[g].append(d.loc[np.concatenate(idx_group[:6])])
        
        for _n, (_g, _d) in enumerate(final_group.items()):
            final_group[_g].append(d.loc[idx_group[6+_n]])
    
    ffinal_group = {}
    for _n, (g, d) in enumerate(final_group.items()):
        dbc = pd.concat(d)
        ffinal_group[f'site{_n+1}'] = [[dbc.iloc[i]['age'], dbc.iloc[i]['sex'], dbc.iloc[i]['ecg'], dbc.iloc[i]['flag']] for i in range(dbc.shape[0])]
    
    with open(f'data/{f}_single.pickle', 'wb') as f:
        pickle.dump(ffinal_group, f, pickle.HIGHEST_PROTOCOL)
