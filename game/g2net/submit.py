
import os
import gc

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
from albumentations import *
from Train import Config
from Model import Model
batch_size = 32
submit = pd.read_csv('/home/wangjingqi/input/dataset/g2net/sample_submission.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS =["/home/wangjingqi/input/ck/g2net/inception_v4"]


import h5py
class Dataset(torch.utils.data.Dataset):
   
    def __init__(self,config, df,tfms=False):
        self.config = config
        self.ids =  df.id.values
        self.tfms = tfms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        x = np.load(os.path.join("/home/wangjingqi/input/dataset/g2net/test",id+".npy"),allow_pickle=True).item()
        h1 = self.process(x["H1"])
        l1 = self.process(x["L1"])
        x = np.concatenate((np.expand_dims(h1,axis=0),np.expand_dims(l1,axis=0)),axis=0)
        x = np.transpose(x,(1,2,0))
        x = np.mean(x.reshape(360,self.config.shape[0],self.config.shape[1],2), axis=2)

        if self.tfms=="horizontal flip":
            x = HorizontalFlip(p=1)(image=x)["image"]
        elif self.tfms=="vertical flip":
            x = VerticalFlip(p=1)(image=x)["image"]
        elif self.tfms=="horizontal and vertical flip":
            x = HorizontalFlip(p=1)(image=x)["image"]
            x = VerticalFlip(p=1)(image=x)["image"]
        elif self.tfms is None:
            pass
        x = np.transpose(x,(2,0,1))
        x=torch.from_numpy(x).float()
        return x
    @classmethod
    def process(cls, data: np.ndarray) -> np.ndarray:
        data = data* 1e22
        data = data.imag**2 + data.real**2
        data = data/data.mean()
        x = np.zeros((360, 4096))
        x[:, :data.shape[-1]] = data[:,:4096]
        return x

def predict(tfms=None):
    preds = np.zeros(len(submit))
    weight_sum = 0
    with torch.no_grad():
        for model_dir in MODELS:
            model_config = Config.from_json_file(os.path.join(model_dir,"config.json"))
            model_config.pretrained = False
            pred_per_model = np.zeros(len(submit))
            for m in os.listdir(model_dir):
                if m.endswith(".pth"):
                    model=Model(model_config)
                    mp = os.path.join(model_dir,m)
                    ck = torch.load(mp,map_location=torch.device('cpu'))
                    model.load_state_dict(state_dict=ck) 
                    dataset = Dataset(model_config,submit,tfms)
                    dataloader = DataLoader(dataset,batch_size =batch_size,num_workers=8, shuffle = False, pin_memory=False,drop_last=False)
                    model.cuda()
                    model.eval()
                    model.float()
                    i = 0 
                    with tqdm(desc=f"model-{tfms}-{m}", unit='it', total=len(dataloader)) as pbar:
                        for batch in dataloader:
                            batch = batch.cuda()
                            pred = torch.sigmoid(model(batch)).detach().cpu().numpy()
                            # pred = model(batch).detach().cpu().numpy()
                            pred_per_model[i:i+len(pred)] += pred
                            i +=len(pred)
                            pbar.update()
            preds +=model_config.weight *pred_per_model/model_config.nfolds
            weight_sum += model_config.weight
            preds = preds/weight_sum
            return preds
preds0 = predict(tfms=None)
preds1 = predict(tfms="horizontal flip")
preds2 = predict(tfms="vertical flip")
preds3 = predict(tfms="horizontal and vertical flip")
preds = (preds0+preds1+preds2+preds3)/4
submit["target"] = preds
submit.to_csv('/home/wangjingqi/input/ck/g2net/inception_v4.csv',index=False)
print('target range [%.2f, %.2f]' % (submit['target'].min(), submit['target'].max()))