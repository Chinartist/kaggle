import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pandas as pd
import torchaudio
import cv2
from albumentations import *
import random
#create dataset

class G2NetDataset(Dataset):
    def __init__(self, config,dir,df,train=True):
        self.config = config
        self.dir = dir
        self.ids =  df.id.values
        self.labels = df.target.values
        self.train = train
        p = 0.5
        self.tfms = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ChannelDropout(channel_drop_range=(1, 1), fill_value=0,p=p),
], p=1)
        self.time_mask_num = 1 # number of time masking
        self.freq_mask_num = 2 # number of frequency masking
        self.transforms_time_mask = nn.Sequential(
                torchaudio.transforms.TimeMasking(time_mask_param=10),
            )

        self.transforms_freq_mask = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
            )
    def __len__(self):
        return len(self.ids)
    @classmethod
    def process(cls, data: np.ndarray) -> np.ndarray:
        data = data* 1e22
        data = data.imag**2 + data.real**2
        data = data/data.mean()
        x = np.zeros((360, 4096))
        x[:, :data.shape[-1]] = data[:,:4096]
        return x
    
    def __getitem__(self, idx):
        #load data
        id = self.ids[idx]
        y = self.labels[idx]
        if os.path.exists(os.path.join(self.dir[0],id+".npy")):
            x = np.load(os.path.join(self.dir[0],id+".npy"),allow_pickle=True).item()
        else:
            x = np.load(os.path.join(self.dir[1],id+".npy"),allow_pickle=True).item()
        #process data
        h1 = x["H1"]
        l1 = x["L1"]
        x = np.concatenate((np.expand_dims(h1,axis=0),np.expand_dims(l1,axis=0)),axis=0)
        #data augmentation
        x = np.transpose(x,(1,2,0))
        # x = np.mean(x.reshape(360,self.config.shape[0],self.config.shape[1],2), axis=2)

        if self.train:
            x = self.tfms(image=x)['image']
            if np.random.rand() <= 1: # vertical shift
                x = np.roll(x, np.random.randint(low=0, high=x.shape[0]), axis=0)
            x = np.transpose(x,(2,0,1))
            x = torch.from_numpy(x)
            if self.config.shape[0] not in [1024]:
                for _ in range(self.time_mask_num): # tima masking
                    x = self.transforms_time_mask(x)
                for _ in range(self.freq_mask_num): # frequency masking
                    x = self.transforms_freq_mask(x)
        else:
            x = np.transpose(x,(2,0,1))
            x = torch.from_numpy(x)
        x = x.float()
        y = torch.tensor(y).float()
        return x,y
