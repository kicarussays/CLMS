import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet1d(nn.Module):
    def __init__(self, lead=12, fc_dim=256):
        super().__init__()
        self.reduce1 = nn.Sequential(
            nn.Conv1d(lead, lead*8, kernel_size=11, stride=2, padding=11//2),
            nn.BatchNorm1d(lead*8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        self.reduce2 = nn.Sequential(
            nn.Conv1d(lead*8, lead*8, kernel_size=7, stride=2, padding=11//2),
            nn.BatchNorm1d(lead*8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        self.reduce3 = nn.Sequential(
            nn.Conv1d(lead*8, lead*8, kernel_size=5, stride=2, padding=11//2),
            nn.BatchNorm1d(lead*8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        self.resblock1 = nn.Sequential(
            nn.Conv1d(lead*8, lead*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*8),
            nn.ReLU(),
            nn.Conv1d(lead*8, lead*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*8)
        )
        
        self.activation = nn.ReLU()

        self.reduce4 = nn.Sequential(
            nn.Conv1d(lead*8, lead*16, kernel_size=5, stride=1, padding=5//2),
            nn.BatchNorm1d(lead*16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        self.resblock2 = nn.Sequential(
            nn.Conv1d(lead*16, lead*16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*16),
            nn.ReLU(),
            nn.Conv1d(lead*16, lead*16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*16)
        )
        
        self.reduce5 = nn.Sequential(
            nn.Conv1d(lead*16, lead*32, kernel_size=5, stride=1, padding=5//2),
            nn.BatchNorm1d(lead*32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        self.resblock3 = nn.Sequential(
            nn.Conv1d(lead*32, lead*32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*32),
            nn.ReLU(),
            nn.Conv1d(lead*32, lead*32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*32)
        )

        self.demo1 = nn.Linear(2, 32)
        self.demo2 = nn.Linear(32, 64)
        self.demo3 = nn.Linear(64, 64)
        
        self.shared1 = nn.Linear(640*lead + 64, fc_dim*2)
        self.shared2 = nn.BatchNorm1d(fc_dim*2)
        
        self.fc2 = nn.Linear(fc_dim*2, fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)
        
        self.fc3 = nn.Linear(fc_dim, 2)
        self.fclayer = nn.Sequential(self.fc2, self.bn2, self.fc3)
        
        
    def forward(self, x, demo, train=True, feature=False):
        out = self.reduce1(x)
        out = self.reduce2(out)
        out = self.reduce3(out)
        
        out = self.activation(self.resblock1(out) + out)
        out = self.activation(self.resblock1(out) + out)
        
        out = self.reduce4(out)
        
        out = self.activation(self.resblock2(out) + out)
        out = self.activation(self.resblock2(out) + out)
        
        out = self.reduce5(out)
        
        out = self.activation(self.resblock3(out) + out)
        if feature: return self.resblock3(out) + out
        out = self.activation(self.resblock3(out) + out)
        
        demo = self.demo1(demo)
        demo = self.demo2(demo)
        demo = self.demo3(demo)
        
        out = torch.cat((out.view(out.size(0), -1), demo), dim=1)
        out = self.shared2(self.shared1(out))
        
        if train:
            return self.fclayer(out)
        else:
            return out

