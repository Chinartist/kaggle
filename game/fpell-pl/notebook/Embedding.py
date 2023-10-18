import argparse
import sys
import pandas as pd
import re
import string
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import AutoModel,AutoTokenizer
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from tqdm import tqdm
import numpy as np
import warnings


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str,default=None)
parser.add_argument('--model_name', type=str,default=None)
parser.add_argument('--feats_name', type=str,default=None)

parser.add_argument('--bs', type=int,default=3)
args = parser.parse_args()

import os
from types import SimpleNamespace
from pathlib import Path

#for notebook
config = SimpleNamespace(**{})
config.train = "../input/feedback-prize-english-language-learning/train.csv"
config.test ="../input/feedback-prize-english-language-learning/test.csv"
config.submit = "../input/feedback-prize-english-language-learning/sample_submission.csv"
config.pretrain = "../input/deberta"

train_df = pd.read_csv(config.train)
train_df["src"] = "train"
test_df = pd.read_csv(config.test)
test_df["src"] = "test"
ss = pd.read_csv(config.submit)
target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions',]

class fpe_dataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        text = self.df.loc[idx,"full_text"]

        return text
class collator():
    def __init__(self,max_len,pretrained_path,) -> None:
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    def __call__(self, data):
            
        X = self.tokenizer(
                data,
                None,max_length=self.max_len,
                padding='max_length',truncation=True,return_tensors="pt")
        return X

def preprocess(text):
    text = text.split(" ")
    while 2>1:
        if text[-1] in [""," ","\n","\r"]:
            text.pop(-1)
        else:
            break
    return " ".join(text)
test_df['full_text'] = test_df['full_text'].apply(preprocess)



import torch 
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel,AutoTokenizer
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error
import numpy as np
from transformers import get_cosine_schedule_with_warmup,AdamW,get_linear_schedule_with_warmup
from copy import deepcopy
import math


class MaxPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,hidden_state,attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        hidden_state[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(hidden_state, 1)[0]
        return max_embeddings

class MeanPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,hidden_state, attention_mask):

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_state.size())
        )
        mean_embeddings = torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
        return mean_embeddings

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start, layer_weights = None):
        super().__init__()
        if layer_start < 0:
            self.layer_start =num_hidden_layers+1+layer_start
        else:
            self.layer_start = layer_start
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - self.layer_start), dtype=torch.float)
            )

    def forward(self, features):

        all_layer_embedding = torch.stack(features)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weighted_average



class deberta(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.bert,attention_probs_dropout_prob=0.,hidden_dropout_prob=0.,output_hidden_states=True)
        hidden_size = self.encoder.config.hidden_size
        num_attention_heads = self.encoder.config.num_attention_heads
        num_hidden_layers =  self.encoder.config.num_hidden_layers
        
        self.WeightedLayerPooling =nn.Sequential(WeightedLayerPooling(num_hidden_layers=num_hidden_layers,layer_start=config.layer_start,))
        self.Pooler = MeanPooling()
        self.head =nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.GELU(),nn.Linear(hidden_size,6))
    def forward(self,input_ids,attention_mask,token_type_ids):

        b,l = input_ids.shape
        x = dict(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        #for deberta
        encoder = self.encoder(**x)
        all_hidden_states =encoder.hidden_states
        hidden_states = self.WeightedLayerPooling(all_hidden_states)
        hidden_states = self.Pooler(hidden_states,attention_mask)#bs,768
        return hidden_states


import os
import gc
from torch.utils.data import DataLoader
def get_embeddings(mp='',df=None,verbose=True):

    ck = torch.load(mp,map_location=torch.device('cpu'))
    model_config = ck["config"]
    pretrained = os.path.join("../input/deberta",os.path.basename(model_config.bert))
    model_config.bert = os.path.join(pretrained,"model")
    model_config.tokenizer=os.path.join(pretrained,"tokenizer")
    model=deberta(model_config)
    model.load_state_dict(state_dict=ck["model"]) 
    
    dataset = fpe_dataset(df)
    dataloader = DataLoader(dataset,batch_size =args.bs, shuffle = False, pin_memory=False,collate_fn = collator(max_len=model_config.max_len,pretrained_path=model_config.tokenizer),drop_last=False)
    model.cuda()
    model.eval()
    model.float()

    all_text_feats = []
    for batch in tqdm(dataloader,total=len(dataloader)):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        token_type_ids = batch["token_type_ids"].cuda()
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        # Normalize the embeddings
        sentence_embeddings = F.normalize(model_output, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings
        all_text_feats.append(sentence_embeddings)
    all_text_feats = torch.cat(all_text_feats,dim=0)
    if verbose:
        print('test embeddings shape',all_text_feats.shape)
    del model,ck,dataloader,dataset
    gc.collect()
    return all_text_feats





model_path = []
all_test_feats = []

model_path=[ os.path.join(args.model_dir,args.model_name+f"_{fold}.ckpt") for fold in range(5)]

print(len(model_path),model_path)

for m in model_path:
    text_feats_folds = []

    text_feats = get_embeddings(m,test_df)
    text_feats_folds.append(text_feats)
    text_feats_folds = torch.stack(text_feats_folds,dim=0).cpu().numpy()
    print(text_feats_folds.shape)

np.save(f"./test-feats/{args.feats_name}.npy",text_feats_folds)