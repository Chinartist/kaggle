
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
from notebook.config import config

warnings.filterwarnings("ignore")


# %% [markdown]
# **Load data**

# %%
df = pd.read_csv(config.train)
df["src"] = "train"
ss = pd.read_csv(config.submit)
target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions',]
print(df.shape)
print(df.head())

# %% [markdown]
# **MultilabelStratifiedKFold**

# %%
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
df['full_text'] = df['full_text'].apply(preprocess)

# %% [markdown]
# **Model**

# %%

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



# %%
import copy
device = 0
import os
from torch.utils.data import DataLoader
def get_embeddings(mp='',df=None,verbose=True):
    if os.path.exists(mp):
        ck = torch.load(mp,map_location=torch.device('cpu'))
        model_config = ck["config"]
    else:
        print("not exist")
        model_config =copy.deepcopy( config)
    # pretrained = os.path.join("../input/deberta",os.path.basename(model_config.bert))
    # model_config.bert = os.path.join(pretrained,"model")
    model_config.tokenizer=model_config.bert#os.path.join(pretrained,"tokenizer")
    model=deberta(model_config)
    if  os.path.exists(mp):
        model.load_state_dict(state_dict=ck["model"]) 
    
    dataset = fpe_dataset(df)
    dataloader = DataLoader(dataset,batch_size =16, shuffle = False, pin_memory=True,num_workers= 8,collate_fn = collator(max_len=model_config.max_len,pretrained_path=model_config.tokenizer),drop_last=False)
    model.to(device)
    model.eval()
    model.float()

    all_text_feats = []
    for batch in tqdm(dataloader,total=len(dataloader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        # Normalize the embeddings
        sentence_embeddings = F.normalize(model_output, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings
        all_text_feats.append(sentence_embeddings)
    all_text_feats = torch.cat(all_text_feats,dim=0)
    if verbose:
        print('Train embeddings shape',all_text_feats.shape)
        
    return all_text_feats

# %%
all_feats = []

# %%
def get_path(m_dirs):
    model_list = []
    for m in m_dirs:
        model_list.append([os.path.join(m,m.split('/')[-1]+f"_{i}.ckpt") for i in range(5)])
    return model_list

m_dirs = ["deberta-v3-large_lr1-1024-5-2022",]
m_dirs = [os.path.join("/home/wangjingqi/input/ck/fpell",i) for i in m_dirs]
model_path =get_path(m_dirs)

print(len(model_path),model_path)

for models in model_path:
    text_feats_folds = []
    for m in models:
        text_feats = get_embeddings(m,df)
        text_feats_folds.append(text_feats)
    text_feats_folds = torch.stack(text_feats_folds,dim=0).cpu().numpy()
    print(text_feats_folds.shape)
    np.save(f"{m.split('_')[0]}_1024.npy",text_feats_folds)




