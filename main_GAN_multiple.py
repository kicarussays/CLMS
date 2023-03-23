# %%
import os
import sys
import numpy as np
from tqdm import tqdm
import logging
import traceback

import torch
from torch.autograd import grad, Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import saveplot_12lead, LoadData, useDataset
from src.p2pmodel import Pulse2pulseGenerator, Pulse2pulseDiscriminator
from src.config import seed



# %%
# Define Parser
import argparse
parser = argparse.ArgumentParser(description='pulse2pulse Training')
parser.add_argument('--device', '-d', type=int, 
                    help='GPU Device Number', default=3)
parser.add_argument('--lr', type=float, 
                    help='Learning rate', default=0.0001)
parser.add_argument('--bs', type=int, 
                    help='Batch size', default=64)
parser.add_argument('--d', type=int, 
                    help='Model dimentionality', default=50)
parser.add_argument('--epoch', type=int, 
                    help='Max epochs', default=2000)
parser.add_argument('--patience', '-p', type=int, 
                    help='Patiences', default=5)
parser.add_argument('--site', type=str, 
                    help='Site')
parser.add_argument('--domain', type=str, 
                    help='Single or Multiple Domain', default='multiple')
parser.add_argument('--vartime', type=int, 
                    help='validation time', default=20)

args = parser.parse_args()

devicenum = args.device
lr = args.lr
bs = args.bs
lead = 12
d = args.d
max_epoch = args.epoch
site = args.site
n_critic = 5
p_coeff = 10


os.makedirs(f'results/{args.domain}/pulse2pulse/', exist_ok=True)
_logpth = f'results/{args.domain}/pulse2pulse/logs/'
if not os.path.exists(_logpth): os.mkdir(_logpth)
logging.basicConfig(filename=os.path.join(_logpth, f'pulse2pulse_{site}_{bs}_{lr}_{d}.log'), 
                    level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True


# %%
# Define Device
logging.debug(f"GPU {devicenum} OK / Max Epoch: {max_epoch} / Learning Rate: {lr} / Batch Size: {bs} / dimensino: {d}") if torch.cuda.is_available() else logging.debug("No GPU")
device = f"cuda:{str(devicenum)}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


train_site_data, local_site_data, global_site_data = LoadData(seed)
_dl = DataLoader(useDataset(global_site_data['cpsc']), batch_size=64, shuffle=True)
for age, sex, ecg, flag in _dl: break

# %%
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.bias.data.fill_(0)
        

def gradients_status(model, flag):
    for p in model.parameters():
        p.requires_grad = flag
        
        
def sample_noise(size):
    z = torch.FloatTensor(size, lead, 5000).to(device)
    z.data.normal_()  # generating latent space based on normal distribution
    return z


class pulse2pulse_GP(object):
    def __init__(self, train_loader, param):
        super(pulse2pulse_GP, self).__init__()        
        self.g_cost = []
        self.train_d_cost = []
        self.train_w_distance = []
        
        self.discriminator = Pulse2pulseDiscriminator(model_size=d, num_channels=lead).to(device)
        self.generator = Pulse2pulseGenerator(model_size=d, num_channels=lead).to(device)

        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)
    
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr)
        
        self.train_loader = train_loader
        self.n_samples_per_batch = len(train_loader)
        self.param = param
                     
    
    def calculate_discriminator_loss(self, real, generated):
        disc_out_gen = self.discriminator(generated)
        disc_out_real = self.discriminator(real)

        alpha = torch.FloatTensor(real.size(0), 1, 1).uniform_(0, 1).to(device)
        alpha = alpha.expand(real.size(0), real.size(1), real.size(2))

        interpolated = (1 - alpha) * real.data + (alpha) * generated.data[:real.size(0)]
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)
        grad_inputs = interpolated
        ones = torch.ones(prob_interpolated.size()).to(device)
        gradients = grad(
            outputs=prob_interpolated,
            inputs=grad_inputs,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # calculate gradient penalty
        grad_penalty = (
            p_coeff
            * ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        )
        assert not (torch.isnan(grad_penalty))
        assert not (torch.isnan(disc_out_gen.mean()))
        assert not (torch.isnan(disc_out_real.mean()))
        cost_wd = disc_out_gen.mean() - disc_out_real.mean()
        cost = cost_wd + grad_penalty
        return cost, cost_wd
    
    def apply_zero_grad(self):
        self.generator.zero_grad()
        self.optimizer_g.zero_grad()

        self.discriminator.zero_grad()
        self.optimizer_d.zero_grad()

    def enable_disc_disable_gen(self):
        gradients_status(self.discriminator, True)
        gradients_status(self.generator, False)

    def enable_gen_disable_disc(self):
        gradients_status(self.discriminator, False)
        gradients_status(self.generator, True)

    def disable_all(self):
        gradients_status(self.discriminator, False)
        gradients_status(self.generator, False)
        
    def train(self):
        path = f'results/{args.domain}/pulse2pulse/saved/'
        if not os.path.exists(path): os.mkdir(path)
        _site, _flag = self.param
        saveparams = f'{_site}_{_flag}_{bs}_{lr}_{d}'
        gan_model_name = path + f'pulse2pulse_{saveparams}.tar'
        print(f'Model Name: {gan_model_name}')
        
        fixed_noise = sample_noise(bs).to(
            device
        )  # used to save samples every few epochs
        
        first_iter = 0
        if os.path.isfile(gan_model_name):
            print('file exists')
            checkpoint = torch.load(gan_model_name, map_location=device)
            self.generator.load_state_dict(checkpoint["generator"])
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])

            first_iter = checkpoint["epoch"] + 1
            
        logging.debug(f'Total Batches: {len(self.train_loader)}')
        
        best_loss = -999999
        patience = 0
        use_loss = 0
        for epoch in range(first_iter, first_iter+max_epoch):
            self.generator.train()
            self.discriminator.train()
            
            Loss_D = []
            Loss_G = []
            
            for iter_indx, (age, sex, ecg, flag) in tqdm(enumerate(self.train_loader)):
                ecg, flag = ecg.to(device), flag.to(device)
                self.enable_disc_disable_gen()
                
                for _ in range(n_critic):
                    real_signal = ecg.view(ecg.size(0), -1, ecg.size(-1))

                    # need to add mixed signal and flag
                    noise = sample_noise(bs)
                    generated = self.generator(noise)
                    #############################
                    # Calculating discriminator loss and updating discriminator
                    #############################
                    self.apply_zero_grad()
                    disc_cost, disc_wd = self.calculate_discriminator_loss(
                        real_signal, generated
                    )
                    assert not (torch.isnan(disc_cost))
                    disc_cost.backward()
                    self.optimizer_d.step()
                    
                    Loss_D.append(disc_cost.item())

                self.disable_all()
                
                self.apply_zero_grad()
                self.enable_gen_disable_disc()
                noise = sample_noise(bs)
                generated = self.generator(noise)
                discriminator_output_fake = self.discriminator(generated)
                generator_cost = -discriminator_output_fake.mean()
                generator_cost.backward()
                self.optimizer_g.step()
                self.disable_all()
                
                Loss_G.append(generator_cost.item() * -1)
                

            self.g_cost.append(generator_cost.item() * -1)
            self.train_d_cost.append(disc_cost.item())
            self.train_w_distance.append(disc_wd.item() * -1)
            
                
            saving_dict = {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "epoch": epoch,
                "optimizer_d": self.optimizer_d.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "train_d_cost": self.train_d_cost,
                "train_w_distance": self.train_w_distance,
                # "valid_g_cost": self.valid_g_cost,
                "g_cost": self.g_cost,
            }
            
            logging.debug('Epoch: [%4d]\tLoss_D: %.4f\tLoss_G: %.4f'
                % (epoch+1, np.mean(Loss_D), np.mean(Loss_G)))
            
            if (epoch + 1) % args.vartime == 0:
                fake = self.generator(fixed_noise).detach().cpu().numpy()
                figpth = f'results/{args.domain}/pulse2pulse/figure/{args.site}/{_flag}'
                saveplot_12lead(figpth, bs, fake, epoch, lead, 'pulse2pulse')
                
                if use_loss / args.vartime > best_loss:
                    logging.debug(f'{use_loss / args.vartime} > {best_loss} Save Best Model...')
                    torch.save(saving_dict, gan_model_name)
                    best_loss = use_loss / args.vartime
                    patience = 0
                else:
                    patience += 1
                
                use_loss = 0
                if patience > args.patience and epoch > 10000 / len(self.train_loader): break
                
            else:
                use_loss += np.mean(Loss_D)
                
                    
if __name__ == "__main__":    
    try:
        flag_group = [0, 1] # 0: Normal 1: Arrhythmia
        
        # for site, data in train_site_data.items():
        data = train_site_data[site]
        for flag in flag_group:
            logging.debug(f'flag {flag} training...')
            usedata = [_d for _d in data if _d[3] == flag]  # age, sex, ecg, flag
            
            train_loader = DataLoader(useDataset(usedata), batch_size=bs, shuffle=True)
            param = (site, flag)
            pulse2pulse = pulse2pulse_GP(train_loader, param)
            pulse2pulse.train()
                        
    except:
        logging.error(traceback.format_exc())

    


# %%
