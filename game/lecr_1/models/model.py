
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModel,AutoTokenizer
from sentence_transformers import SentenceTransformer
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error
import numpy as np
from transformers import get_cosine_schedule_with_warmup,AdamW,get_linear_schedule_with_warmup
from copy import deepcopy
import math
from .utils import MeanPooling,WeightedLayerPooling,AttentionPooling
def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)
class model(pl.LightningModule):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained("/mnt/hdd1/wangjingqi/ck/lecr/ft/pmmb2_TiDeTe/26250",attention_probs_dropout_prob=cfg.drop,hidden_dropout_prob=cfg.drop,output_hidden_states=True)
        num_hidden_layers =  self.encoder.config.num_hidden_layers
        self.WeightedLayerPooling =nn.Sequential(WeightedLayerPooling(num_hidden_layers=num_hidden_layers,layer_start=cfg.layer_start,))
        if cfg.pool == "mean":
            self.Pooler = MeanPooling()
        elif cfg.pool == "attention":
            self.Pooler = AttentionPooling(self.encoder.config.hidden_size)
        self.fc = nn.Linear(self.encoder.config.hidden_size,1)
        self._init_weights(self.fc)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def forward_features(self,input_ids,attention_mask):
        x = dict(input_ids=input_ids,attention_mask=attention_mask)
        encoder = self.encoder(**x)
        all_hidden_states =encoder.hidden_states
        hidden_states = self.WeightedLayerPooling(all_hidden_states)
        hidden_states = self.Pooler(hidden_states,attention_mask)
        return hidden_states
    def forward(self,inputs):
        hidden_states = self.forward_features(**inputs)
        outputs = self.fc(hidden_states)
        return outputs

class Model(model):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)
        self.reinit_weight()

    def reinit_weight(self,):
        reinit_layers = self.cfg.reinit_layers 
        if reinit_layers > 0:
            print(f'Reinitializing Last {reinit_layers} Layers ...')
            encoder_temp = self.encoder
            for layer in encoder_temp.encoder.layer[-reinit_layers:]:
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
                        if module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
        return None

    def loss(self,preds,targets):
        l = nn.BCEWithLogitsLoss(reduction = "mean")(preds,targets)
        return l

    def training_step(self,batch,batch_idx):
        inputs = batch['inputs']
        targets = batch["labels"]
        preds = self(inputs)
        loss = self.loss(preds.squeeze(-1),targets)
        return {'loss': loss}

    def validation_step(self,batch,batch_idx):
        inputs = batch['inputs']
        targets = batch["labels"]
        with torch.no_grad():
            preds = self(inputs)
            preds = preds.squeeze(-1)
        return {"preds":preds,"targets":targets}

    def validation_epoch_end(self,outputs):
        preds = torch.cat([x['preds'] for x in outputs],dim=0)
        targets = torch.cat([x['targets'] for x in outputs],dim=0)
        preds = preds.sigmoid().cpu().numpy()
        score, threshold = self.get_best_threshold(self.cfg.x_val, preds, self.cfg.correlations)
        self.log('scores', score , prog_bar=True,sync_dist=True)
        self.log('threshold', threshold , prog_bar=True,sync_dist=True)
        return None

    def get_best_threshold(self,x_val, val_predictions, correlations):
        best_score = 0
        best_threshold = None
        for thres in np.arange(0.001, 0.1, 0.001):
            x_val['predictions'] = np.where(val_predictions > thres, 1, 0)
            x_val1 = x_val[x_val['predictions'] == 1]
            x_val1 = x_val1.groupby(['topics_ids'])['content_ids'].unique().reset_index()
            x_val1['content_ids'] = x_val1['content_ids'].apply(lambda x: ' '.join(x))
            x_val1.columns = ['topic_id', 'predictions']
            x_val0 = pd.Series(x_val['topics_ids'].unique())
            x_val0 = x_val0[~x_val0.isin(x_val1['topic_id'])]
            x_val0 = pd.DataFrame({'topic_id': x_val0.values, 'predictions': ""})
            x_val_r = pd.concat([x_val1, x_val0], axis = 0, ignore_index = True)
            x_val_r = x_val_r.merge(correlations, how = 'left', on = 'topic_id')
            score = f2_score(x_val_r['content_ids'], x_val_r['predictions'])
            if score > best_score:
                best_score = score
                best_threshold = thres
        return best_score, best_threshold

    def configure_optimizers(self):
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters()
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.cfg.lr,eps=1e-6,correct_bias=True)
        epoch_steps = self.cfg.data_len
        batch_size = self.cfg.bs
        epochs = self.cfg.epochs
        training_steps = epochs* epoch_steps // batch_size
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = training_steps* self.cfg.warmup_ratio, 
            num_training_steps = training_steps, 
            num_cycles = self.cfg.num_cycles
            )
        lr_scheduler_cfg = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency':  1,
            }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_cfg}


    def get_optimizer_grouped_parameters(self, 
    ):
        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if "encoder" not in n ],
                "weight_decay": 0.0,
                "lr": self.cfg.lr,
            }
        ]
        # initialize lrs for every layer
        layers = [self.encoder.embeddings] + list(self.encoder.encoder.layer)
        layers.reverse()
        lr = self.cfg.lr
        wd = self.cfg.wd
        for i,layer in enumerate(layers):
            if i%self.cfg.llrd_interval==0:
                    lr *= self.cfg.llrd
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": wd,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
        
            
        return optimizer_grouped_parameters







    


