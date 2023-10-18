
from notebook.config import config
import pandas as pd
from fastai.vision.all import *
import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np
from transformers import AutoModel,AutoTokenizer
from tqdm import tqdm
class fpe_dataset(torch.utils.data.Dataset):
    def __init__(self,df,train=True):
        self.train = train
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        
        target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions',]
        
 
        text = self.df.loc[idx,"full_text"]

        labels = torch.tensor(self.df.loc[idx,target_cols].values.tolist()).float()
    
        X = {"text":text,"labels":labels}

        return X

class collator():
    def __init__(self,max_len,pretrained_path,) -> None:
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    def __call__(self, data):
        texts = [x["text"] for x in data]
            
        X = self.tokenizer(
                texts,
                None,max_length=self.max_len,
                padding='max_length',truncation=True,return_tensors="pt")
        labels = torch.stack([x["labels"] for x in data])
        X.update({"labels":labels})
        return X
 

def preprocess(text):
    text = text.split(" ")
    while 2>1:
        if text[-1] in [""," ","\n","\r"]:
            text.pop(-1)
        else:
            break
    return " ".join(text)