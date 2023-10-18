import os
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping,StochasticWeightAveraging
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger,CSVLogger
import warnings
from sklearn.model_selection import KFold
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
from transformers import PretrainedConfig
from lecr_dataset import LECRDataset,collator
from models.model import Model
from sklearn.model_selection import StratifiedGroupKFold
import string
import re

class Config(PretrainedConfig):
    def __init__(self,
    model_name="xlm-roberta-base",
    save_name ="xlm-roberta-base",
    bs=384,lr=1.2e-4,wd=0.001,llrd_interval=1.0,llrd=0.9,
    epochs = 6,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    seed = 666,
    nfolds = 5,
    num_workers = 8,
    device_ids = [3],
    used_folds = [3],
    weight = 1.0,
    drop = 0.,
    val_check_interval=1.0,
    precision = 16,
    reinit_layers=1,
    layer_start = -1,
    num_cycles = 0.5,
    warmup_ratio = 0.1,
    swa_lr = 6e-5,
    pool="mean",
    train = "/mnt/hdd1/wangjingqi/dataset/lecr/train_50_pmmb2_TiDeTe_26250_TiDeTe_TiDe.csv",
    max_len = 80,
    len1 = 32,
    len2 = 54,
    transformers_version=None,
    ):
        self.model_name = model_name
        self.bs = bs
        self.lr = lr
        self.wd = wd
        self.llrd_interval=llrd_interval
        self.llrd= llrd
        self.epochs = epochs
        self.seed = seed
        self.nfolds = nfolds
        self.num_workers = num_workers
        self.device_ids = device_ids
        self.used_folds = used_folds
        self.weight = weight
        self.drop = drop
        self.val_check_interval = val_check_interval
        self.precision = precision
        self.reinit_layers = reinit_layers
        self.layer_start = layer_start
        self.num_cycles = num_cycles
        self.warmup_ratio = warmup_ratio
        self.swa_lr = swa_lr
        self.pool = pool
        self.train = train
        self.max_len = max_len
        self.save_name = save_name+"_"+train.split("/")[-1].split(".")[0]+"_split"+f"{len1}_{len2}_super"
        self.len1 = len1
        self.len2 = len2

def main():
    cfg = Config()
    print(cfg)
    os.makedirs(os.path.join("/mnt/hdd1/wangjingqi/ck/lecr",cfg.save_name),exist_ok=True)
    os.makedirs("/mnt/hdd1/wangjingqi/lecr_1/log",exist_ok=True)
    
    # cfg.to_json_file(os.path.join("/mnt/hdd1/wangjingqi/ck/lecr",cfg.save_name,"config.json"))
    seed_everything(cfg.seed,workers=True)
    train = pd.read_csv(cfg.train)
    correlations = pd.read_csv('/mnt/hdd1/wangjingqi/dataset/lecr/correlations.csv')
    # Create feature column
    train['input1'].fillna("", inplace = True)
    train['input2'].fillna("", inplace = True)
    train["input1"] = train["input1"].apply(lambda x: " ".join(x.split()[:cfg.len1-1]))
    train["input2"] = train["input2"].apply(lambda x: " ".join(x.split()[:cfg.len2-2]))

    train['input'] = train['input1'] + '[SEP]' + train['input2']
    print(' ')
    print('-' * 50)
    print(f"train.shape: {train.shape}")
    print(f"correlations.shape: {correlations.shape}")


    kfold = StratifiedGroupKFold(n_splits = cfg.nfolds, shuffle = True, random_state = cfg.seed)
    for num, (train_index, val_index) in enumerate(kfold.split(train, train['target'], train['topics_ids'])):
        train.loc[val_index, 'fold'] = int(num)
    train.to_csv("/mnt/hdd1/wangjingqi/lecr_1/train.csv",index=False)
    train['fold'] = train['fold'].astype(int)
    for fold in cfg.used_folds:
        print(' ')
        print(f"========== fold: {fold} training ==========")
        # Split train & validation
        x_train = train[train['fold'] != fold]
        x_val = train[train['fold'] == fold]
        train_dataset = LECRDataset(cfg,x_train)
        val_dataset = LECRDataset(cfg,x_val)
        cfg.data_len = len(train_dataset)
        cfg.x_val = x_val
        cfg.correlations = correlations
        save_name = cfg.save_name+f"_{fold}"


        train_dataloader = DataLoader(train_dataset, batch_size = cfg.bs, shuffle = True, num_workers= 4, pin_memory=True,collate_fn = collator(cfg),drop_last=True)
        validation_dataloader = DataLoader(val_dataset,batch_size = cfg.bs, shuffle = False, num_workers= 4, pin_memory=True,collate_fn = collator(cfg),drop_last=False)
        early_stop_callback = EarlyStopping(monitor="scores", min_delta=0.00, patience=5, verbose= False, mode="max")
        checkpoint_callback = ModelCheckpoint(monitor='scores',
                                          dirpath= os.path.join("/mnt/hdd1/wangjingqi/ck/lecr",cfg.save_name),
                                      save_top_k=1,
                                      save_last= False,
                                      save_weights_only=True,
                                      filename= save_name,
                                      verbose= True,
                                      mode='max')
        model = Model(cfg=cfg)
        swa_callback = StochasticWeightAveraging(swa_epoch_start=3, swa_lrs=cfg.swa_lr,annealing_epochs=cfg.epochs)
        
        print("Model Creation")
        trainer_cfg=dict(
                logger=CSVLogger(save_dir="/mnt/hdd1/wangjingqi/lecr_1/log",name=cfg.save_name,version=save_name),
                max_epochs= cfg.epochs,
                min_epochs =cfg.epochs*2//3 ,
                callbacks=[checkpoint_callback,early_stop_callback
                ],
                sync_batchnorm=True,
                num_sanity_val_steps=0,
                gpus = cfg.device_ids,
                accelerator="gpu",
                strategy=DDPStrategy(find_unused_parameters=True),
                val_check_interval=cfg.val_check_interval,
                precision=cfg.precision, 
        )
        trainer = Trainer(**trainer_cfg)    
        trainer.fit(model , train_dataloader , validation_dataloader)

if __name__ == "__main__":
    main()