from tokenize import group
from transformers import BertModel, BertTokenizer,DistilBertTokenizerFast
from scipy.stats import spearmanr

import pandas as pd
from fastai.vision.all import *
import torch
from torch.utils.data import DataLoader,Dataset
from collections import Counter, OrderedDict
import numpy as np
import Levenshtein 
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ====================================================
# Dataset
# ====================================================
def get_context(cfg, inputs, position):
    for i in range(position-cfg.context_size, position+cfg.context_size+1):
        if i >= 0 and i < len(inputs):
            inputs[i] = 1
        elif i >= len(inputs):
            break
    return torch.tensor(inputs, dtype=torch.long)
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=cfg.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts1 = df['sequence'].values
        self.texts2 = df['mutant_seq'].values
        self.labels = df[cfg.target_cols].values
        self.position = df['position'].values

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, item):
        inputs1 = prepare_input(self.cfg, self.texts1[item])
        inputs2 = prepare_input(self.cfg, self.texts2[item])
        position = np.zeros(self.cfg.max_len)
        position[self.position[item]] = 1

        position = torch.tensor(position, dtype=torch.int)
        context_mask = get_context(self.cfg, np.zeros(self.cfg.max_len), self.position[item])*inputs1["attention_mask"]
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs1, inputs2, position,context_mask, label
    
