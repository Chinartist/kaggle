
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
from .models import MeanPooling,WeightedLayerPooling
class deberta(pl.LightningModule):
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
        pool_out = self.head(hidden_states)
        outs =pool_out
        return outs

class Model(deberta):
    def __init__(self,config) -> None:
        super().__init__(config)
   
    def init_weight(self,init_list,fold):
        for module in self.head.modules():
            if isinstance(module,nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
        reinit_layers = self.config.reinit_layers 
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
    
    def freeze(self,config):
        if config.freeze is not  None:
            for n,p in self.encoder.named_parameters():
                if config.freeze in n :
                    break
                p.requires_grad = False
               
            print("freeze",config.freeze)
    def comp_score(self,y_true,y_pred):
        rmse_scores = []
        for i in range(y_pred.shape[-1]):
            rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i])))
        return np.mean(rmse_scores)
    def loss(self,outputs,targets):
        l = nn.SmoothL1Loss()(outputs,targets)
        return l
    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids =batch['token_type_ids']
        target = batch['labels'] 
        outputs = self(input_ids,attention_mask,token_type_ids)
        output = outputs

        loss = self.loss(output,target)

        return {'loss': loss}

    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids =batch['token_type_ids']
        target = batch['labels'] 

        with torch.no_grad():
            output = self(input_ids,attention_mask,token_type_ids)

        loss = self.loss(output,target)
        
        return {'val_loss': loss, 'logits': output,'targets':target}   

    def validation_epoch_end(self,outputs):

        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', val_loss, prog_bar=True,sync_dist=True)

        output_val = torch.cat([x['logits'] for x in outputs],dim=0).cpu().detach().numpy()
        target_val = torch.cat([x['targets'] for x in outputs],dim=0).cpu().detach().numpy()
        avg_score = self.comp_score(output_val,target_val)
        target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions',]
        for i,targ in enumerate(target_cols):
            score = self.comp_score(output_val[:,i:i+1],target_val[:,i:i+1])
            self.log(targ,score, prog_bar=False,sync_dist=True)
        self.log('val_mcrmse', avg_score , prog_bar=True,sync_dist=True)
        return None

    def configure_optimizers(self):
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters()
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.config.head_lr,eps=1e-6,correct_bias=True)

        epoch_steps = self.config.data_len
        batch_size = self.config.bs
        epochs = self.config.epochs
        
        training_steps = epochs* epoch_steps // batch_size
        # scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training_steps)
      
        lr_scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency':  1,
            }
       

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
    def get_optimizer_grouped_parameters(self, 
    ):
        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if "head" in n ],
                "weight_decay": 0.0,
                "lr": self.config.head_lr,
            },
            {
                "params": [p for n, p in self.named_parameters() if  "WeightedLayerPooling" in n],
                "weight_decay": 0.0,
                "lr": self.config.head_lr,
            },
            {
                "params": [p for n, p in self.named_parameters() if  "Pooler" in n],
                "weight_decay": 0.0,
                "lr": self.config.head_lr,
            },
        ]
        
        # initialize lrs for every layer
        layers = [self.encoder.embeddings] + list(self.encoder.encoder.layer)
        layers.reverse()
        lr = self.config.head_lr
        wd = self.config.wd
        for i,layer in enumerate(layers):
            if i%self.config.llrd_interval==0:
                    lr *= self.config.llrd
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







    


