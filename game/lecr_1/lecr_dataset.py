
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np
from transformers import AutoModel,AutoTokenizer
from tqdm import tqdm
import random
class LECRDataset(torch.utils.data.Dataset):
    def __init__(self,cfg,df):
        self.cfg = cfg
        self.inputs = df.input.values
        self.labels = df.target.values
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,idx):
        
        sample = {"input":self.inputs[idx],"label":self.labels[idx]}
        return sample
class collator():
    def __init__(self,cfg) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.cfg = cfg
    def __call__(self, data):
        inputs = []
        labels = []
        for sample in data:
            inputs.append(sample["input"])
            labels.append(sample["label"])
        inputs = self.tokenize(inputs)
        labels = torch.tensor(labels).float()
        return {"inputs":inputs,"labels":labels}
    def tokenize(self,texts):
            return self.tokenizer(
                texts,padding="longest",max_length=self.cfg.max_len,truncation=True,return_tensors="pt")
