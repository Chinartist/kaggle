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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils.muti_init import xavier_uniform_init,xavier_normal_init,he_init,kiming_init,orthogonal_init
from notebook.config import config
from Models.deberta import Model
from fpe_dataset import fpe_dataset,collator,preprocess

import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

seed_everything(config.seed,workers=True)

config.ck = os.path.join(config.ck,config.model_fname)
os.makedirs(config.ck,exist_ok=True)
os.makedirs(config.log,exist_ok=True)

def main():
    
    train_df = pd.read_csv(config.train)
    train_df['full_text'] = train_df['full_text'].apply(preprocess)
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions',]
    train_df.loc[:,"seq_len"] = train_df.full_text.apply(lambda x: len(x.split(" ")))
    train_df.loc[:,"n_unique"] = train_df.full_text.apply(lambda x: len(np.unique(x.split(" "))))
    train_df.loc[:,"avg_score"] = np.average(train_df.loc[:,target_cols],axis=1)
    skf = MultilabelStratifiedKFold(n_splits=config.nfolds, shuffle=True, random_state=config.seed)
    split_cols =target_cols

    for i,(train_index,val_index) in enumerate(skf.split(train_df,train_df[split_cols])):
        train_df.loc[val_index,"fold"] = i


    
    init_list = [xavier_normal_init]

    for fold in config.used_folds:

        train_ =train_df[train_df.fold != fold].reset_index(drop = True)
        val_ = train_df[train_df.fold==fold].reset_index(drop = True)

        train_dataset = fpe_dataset(train_,train=True)
        val_dataset = fpe_dataset(val_,train=False)

        model_fname = config.model_fname+f"_{fold}"

        train_dataloader = DataLoader(train_dataset, batch_size = config.bs, shuffle = True, num_workers= 8, pin_memory=True,collate_fn = collator(max_len=config.max_len,pretrained_path=config.bert,),drop_last=True)
        validation_dataloader = DataLoader(val_dataset,batch_size = config.bs, shuffle = False, num_workers= 8, pin_memory=True,collate_fn = collator(max_len=config.max_len,pretrained_path=config.bert,),drop_last=False)
        
        early_stop_callback = EarlyStopping(monitor="val_mcrmse", min_delta=0.00, patience=config.patience, verbose= False, mode="min")
        checkpoint_callback = ModelCheckpoint(monitor='val_mcrmse',
                                          dirpath= config.ck,
                                      save_top_k=1,
                                      save_last= False,
                                      save_weights_only=True,
                                      filename= model_fname,
                                      verbose= True,
                                      mode='min')

        config.data_len = len(train_dataset)
        model = Model(config=config)
        model.freeze(config=config)
        model.init_weight(init_list,fold)
        print("Model Creation")

        trainer_cfg=dict(
                logger=CSVLogger(save_dir=config.log,name=config.model_fname,version=model_fname),
                max_epochs= config.epochs,
                min_epochs =config.epochs*2//3 ,
                callbacks=[checkpoint_callback
                ],
                sync_batchnorm=True,
                num_sanity_val_steps=0,
                gpus = config.device_ids,
                accelerator="gpu",
                strategy=DDPStrategy(find_unused_parameters=True),
                val_check_interval=config.val_check_interval,
                precision=config.precision, 
                accumulate_grad_batches = config.accumulate_grad_batches,
        )
        
        trainer = Trainer(**trainer_cfg)    
        trainer.fit(model , train_dataloader , validation_dataloader)

        print(config)

if __name__ == "__main__":
    main()

    