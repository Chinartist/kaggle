


from ast import While
from fastai.vision.all import *
from fastai.callback.tensorboard import TensorBoardCallback
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import cv2
import gc
import random
from albumentations import *
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from config import config
class Dice_all(Metric):#
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        y = 1*(learn.y>0)
        preds = learn.pred
        outputs = preds[0]#bs,6,768,768
        organs =preds[2]#bs
        outputs =torch.stack([ torch.stack([outputs[i][0],outputs[i][organs[i]]],dim=0) for i in range(organs.shape[0])],dim=0)
        outputs = torch.softmax(outputs,dim=1)
        mask = torch.argmax(outputs,dim=1)
        outputs = mask
        pred,targ = flatten_check(outputs.unsqueeze(1),y)
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None
class Dice_kidney(Metric):#
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        y = 1*(learn.y>0).squeeze(1)
        preds = learn.pred
        outputs = preds[0]#bs,6,768,768
        organs =preds[2]#bs
        outputs = torch.softmax(outputs,dim=1)
        tmp = []
        y_tmp = []
        for i in range(organs.shape[0]):
            if organs[i] == 1:
                tmp.append(torch.stack([outputs[i][0],outputs[i][organs[i]]],dim=0))
                y_tmp.append(y[i])
        
        if tmp == []:
            self.inter +=0
        else:
            outputs =torch.stack(tmp,dim=0)
            y =torch.stack(y_tmp,dim=0).unsqueeze(1)

            outputs = torch.softmax(outputs,dim=1)
            mask = torch.argmax(outputs,dim=1)
            outputs = mask
            pred,targ = flatten_check(outputs.unsqueeze(1),y)
            self.inter += (pred*targ).float().sum().item()
            self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None
class Dice_prostate(Metric):#
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        y = 1*(learn.y>0).squeeze(1)
        preds = learn.pred
        outputs = preds[0]#bs,6,768,768
        organs =preds[2]#bs
        outputs = torch.softmax(outputs,dim=1)
        tmp = []
        y_tmp = []
        for i in range(organs.shape[0]):
            if organs[i] == 2:
                tmp.append(torch.stack([outputs[i][0],outputs[i][organs[i]]],dim=0))
                y_tmp.append(y[i])
        
        if tmp == []:
            self.inter +=0
        else:
            outputs =torch.stack(tmp,dim=0)
            y =torch.stack(y_tmp,dim=0).unsqueeze(1)

            outputs = torch.softmax(outputs,dim=1)
            mask = torch.argmax(outputs,dim=1)
            outputs = mask
            pred,targ = flatten_check(outputs.unsqueeze(1),y)
            self.inter += (pred*targ).float().sum().item()
            self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None            
class Dice_largeintestine(Metric):#
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        y = 1*(learn.y>0).squeeze(1)
        preds = learn.pred
        outputs = preds[0]#bs,6,768,768
        organs =preds[2]#bs
        outputs = torch.softmax(outputs,dim=1)
        tmp = []
        y_tmp = []
        for i in range(organs.shape[0]):
            if organs[i] == 3:
                tmp.append(torch.stack([outputs[i][0],outputs[i][organs[i]]],dim=0))
                y_tmp.append(y[i])
        
        if tmp == []:
            self.inter +=0
        else:
            outputs =torch.stack(tmp,dim=0)
            y =torch.stack(y_tmp,dim=0).unsqueeze(1)

            outputs = torch.softmax(outputs,dim=1)
            mask = torch.argmax(outputs,dim=1)
            outputs = mask
            pred,targ = flatten_check(outputs.unsqueeze(1),y)
            self.inter += (pred*targ).float().sum().item()
            self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None
class Dice_spleen(Metric):#
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        y = 1*(learn.y>0).squeeze(1)
        preds = learn.pred
        outputs = preds[0]#bs,6,768,768
        organs =preds[2]#bs
        outputs = torch.softmax(outputs,dim=1)
        tmp = []
        y_tmp = []
        for i in range(organs.shape[0]):
            if organs[i] == 4:
                tmp.append(torch.stack([outputs[i][0],outputs[i][organs[i]]],dim=0))
                y_tmp.append(y[i])
        
        if tmp == []:
            self.inter +=0
        else:
            outputs =torch.stack(tmp,dim=0)
            y =torch.stack(y_tmp,dim=0).unsqueeze(1)

            outputs = torch.softmax(outputs,dim=1)
            mask = torch.argmax(outputs,dim=1)
            outputs = mask
            pred,targ = flatten_check(outputs.unsqueeze(1),y)
            self.inter += (pred*targ).float().sum().item()
            self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None                      
class Dice_lung(Metric):#
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        y = 1*(learn.y>0).squeeze(1)
        preds = learn.pred
        outputs = preds[0]#bs,6,768,768
        organs =preds[2]#bs
        outputs = torch.softmax(outputs,dim=1)
        tmp = []
        y_tmp = []
        for i in range(organs.shape[0]):
            if organs[i] == 5:
                tmp.append(torch.stack([outputs[i][0],outputs[i][organs[i]]],dim=0))
                y_tmp.append(y[i])
        if tmp == []:
            self.inter +=0
        else:
            outputs =torch.stack(tmp,dim=0)
            y =torch.stack(y_tmp,dim=0).unsqueeze(1)

            outputs = torch.softmax(outputs,dim=1)
            mask = torch.argmax(outputs,dim=1)
            outputs = mask
            pred,targ = flatten_check(outputs.unsqueeze(1),y)
            self.inter += (pred*targ).float().sum().item()
            self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None
class Dice_soft(Metric):
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        pred,targ = flatten_check(torch.sigmoid(learn.pred[0]), learn.y)
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None

ths =np.arange(0.1,0.9,0.05)
for o in range(5):
    for i in ths:
        config.count[o][i] = 0
# score_dict = {0:0.88,1:0.88,2:0.88,3:0.88,4:0.2}
# dice with automatic threshold selection
class Dice_mutith(Metric):
    def __init__(self, ths=ths
    , axis=1): 
        self.axis = axis
        self.ths = ths
        
    def reset(self): 
        self.inter = torch.zeros(5,len(self.ths))
        self.union = torch.zeros(5,len(self.ths))
        self.organ_rate = torch.zeros(5)
    # def accumulate(self, learn):
    #     if isinstance(learn.pred[0],tuple) or isinstance(learn.pred[0],list):
    #         preds = learn.pred[0][-1]
    #     else:
    #         preds = learn.pred[0]
    #     pred,targ = flatten_check(torch.sigmoid(preds), learn.y)
    #     for i,th in enumerate(self.ths):
    #         p = (pred > th).float()
    #         self.inter[i] += (p*targ).float().sum().item()
    #         self.union[i] += (p+targ).float().sum().item()
    def accumulate(self, learn):
        if isinstance(learn.pred[0],tuple) or isinstance(learn.pred[0],list):
            preds = learn.pred[0][-1]
        else:
            preds = learn.pred[0]
        preds = torch.sigmoid(preds)
        organs = learn.pred[-1]
        # pred,targ = flatten_check(preds, learn.y)
        for j,o in enumerate (organs):
            organ_id = o-1
            self.organ_rate[o-1]+=1
            pred,targ = flatten_check(preds[j],learn.y[j])
            for i,th in enumerate(self.ths):
                p = (pred > th).float()
                self.inter[organ_id][i] += (p*targ).float().sum().item()
                self.union[organ_id][i] += (p+targ).float().sum().item()
    @property
    def value(self):
        dices = 0
        for i in range(5):
            dice = torch.where(self.union[i] > 0.0,2.0*self.inter[i]/self.union[i], torch.zeros_like(self.union[i]))
            config.count[i][self.ths[torch.argmax(dice).item()]] += 1
            dices += dice.max()*self.organ_rate[i]/self.organ_rate.sum()

        return dices
class Dice_th(Metric):
    def __init__(self, ths=np.arange(0.1,0.9,0.05)
    , axis=1): 
        self.axis = axis
        self.ths = ths
        
    def reset(self): 
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))
        
    def accumulate(self, learn):
        if isinstance(learn.pred[0],tuple) or isinstance(learn.pred[0],list):
            preds = learn.pred[0][-1]
        else:
            preds = learn.pred[0]
        pred,targ = flatten_check(torch.sigmoid(preds), learn.y)
        for i,th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p*targ).float().sum().item()
            self.union[i] += (p+targ).float().sum().item()
    @property
    def value(self):
        dices = torch.where(self.union > 0.0, 
                2.0*self.inter/self.union, torch.zeros_like(self.union))
        return dices.max()
from fastai.optimizer import SGD, Adam, QHAdam, OptimWrapper

mean = np.array(config.mean )
std = np.array(config.std)

NUM_WORKERS = config.NUM_WORKERS

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #the following line gives ~10% speedup
    #but may lead to some stochasticity in the results 
    torch.backends.cudnn.benchmark = True
    
seed_everything(config.seed)



def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))
from data_aug import get_p,dropout,get_aug,get_valaug,CutOut,selfcopypaste



class HuBMAPDataset(Dataset):
    def __init__(self, fold, train=True, tfms=None):

        self.p = get_p([0.3,0.5,0.7])
        self.tfms = tfms

        ids = pd.read_csv(config.LABELS,).id.astype(str).values
        organs = pd.read_csv(config.LABELS,).organ.astype(str).values
        self.id2organ = dict(zip(ids,organs))
        kf = KFold(n_splits=config.nfolds,random_state=config.seed,shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        self.organ2id = collections.defaultdict(list)
        for i,o in self.id2organ.items():
            if i in ids:
                self.organ2id[o].append(i)
        self.fnames = [fname for fname in os.listdir(config.TRAIN[0]) if fname.split('.')[0] in ids]
        self.train = train
        self.fold = fold
    def __len__(self):
        return len(self.fnames)
    def get_image(self, fname,i):
      
        img = cv2.cvtColor(cv2.imread(os.path.join(config.TRAIN[i],fname)), cv2.COLOR_BGR2RGB)
        if os.path.exists(os.path.join(config.MASKS[1],fname)):
            mask = cv2.imread(os.path.join(config.MASKS[1],fname),cv2.IMREAD_GRAYSCALE)
        else:
            mask = cv2.imread(os.path.join(config.MASKS[0],fname),cv2.IMREAD_GRAYSCALE)
        return img,mask

    def __getitem__(self, idx):
        fname = self.fnames[idx]

        i = 0
        if random.random() < 0.5:
            i = 1


        img, mask = self.get_image(fname,i)
       
        # if self.train:
        #     img, mask = selfcopypaste(img, mask,fold=self.fold)
        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']
            if random.random() < 0.2 and self.train:
                img,mask = CutOut(img,mask)
 
        return img2tensor((img/255.0 - mean)/std),torch.tensor(config.organ2label[self.id2organ[fname.split('.')[0]]]),img2tensor(mask)
    




def collate_fn(data):
    images,organs,masks =[],[],[]
    for i,o,m in data:
        images.append(i)
        organs.append(o)
        masks.append(m)

    return ((torch.stack(images,dim=0),torch.stack(organs,dim=0)),torch.stack(masks,dim=0))


from new_loss import new_loss
from norm_loss import norm_loss

def symmetric_lovasz(outputs, targets): 

    
    if isinstance(outputs,tuple) or isinstance(outputs,list):
        loss =  norm_loss
        loss0 = 0
        loss1 = 0
        if isinstance(outputs[0],tuple) or isinstance(outputs[0],list):
            for out0 in outputs[0]:
                loss0 += loss(out0,targets)/len(outputs[0])
        else:
            loss0 += loss(outputs[0],targets)
        
        for output in outputs[1]:
            target = F.interpolate(targets,size=output.shape[-2:], mode='nearest')
            loss1 += loss(output,target)
        return loss0+0.2*loss1


def muticlass(outputs, targets):

    
    if isinstance(outputs,tuple) or isinstance(outputs,list):
        loss0 = F.cross_entropy(outputs[0],targets.squeeze(1).long())
      
        return loss0



import pickle

for fold in config.folds:
    
    
    ds_t = HuBMAPDataset(fold=fold, train=True, tfms=get_aug())
    ds_v = HuBMAPDataset(fold=fold, train=False,tfms=get_valaug())

    
    model_fname = f"{config.num_classes}_{config.image_size}_{config.model_name}_{fold}_2021"
    print(config)
    print("used folds:",config.folds)
    print(model_fname)
    print("样本数：",len(ds_t),len(ds_v))

    data = ImageDataLoaders.from_dsets(ds_t,ds_v,bs=config.bs,
                num_workers=NUM_WORKERS,pin_memory=True,collate_fn=collate_fn).to(config.device_ids[0])
    model,split_layers =config.models[config.model_name.split("-")[0]]
    model = model(config=config,fold=fold)

    #model init


    try:
        model.learners.apply(config.init[fold%len(config.init)])
    except:
        pass
    try:
        model.FPN.apply(config.init[fold%len(config.init)])
    except:
        pass
    try:
        model.aux.apply(config.init[fold%len(config.init)])
    except:
        pass
    try:
        model.head.apply(config.init[fold%len(config.init)])
    except:
        pass
    try:
        model.logit.apply(config.init[fold%len(config.init)])
    except:
        pass
    
  
   
    
    if len(config.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)
    model.to(config.device_ids[0])


    loss = symmetric_lovasz
    metrics = [Dice_th(),Dice_mutith()]
    monitor = ["dice_th","dice_mutith"][0]
    log =  pd.DataFrame(columns=["train_loss ",  "valid_loss" ,"dice_th","dice_mutith"])
    

    if config.num_classes>2:
        loss = muticlass
        metrics = [Dice_kidney(),Dice_prostate(),Dice_largeintestine(),Dice_spleen(),Dice_lung(),Dice_all()]
        monitor = "dice_all"
        log = pd.DataFrame(columns=["train_loss ",  "valid_loss" , "dice_kidney"  ,"dice_prostate" , "dice_largeintestine",  "dice_spleen",  "dice_lung",  "dice_all"])
    learn = Learner(data, model, loss_func=loss,
                metrics =metrics ,
                splitter=split_layers).to_fp16()
    #start with training the head
    learn.freeze_to(-1) #doesn't work
    for param in learn.opt.param_groups[0]['params']:
        param.requires_grad = False
    learn.fit_one_cycle(config.head_epoch, lr_max=config.head_lr_max )

    #continue training full model
    learn.unfreeze()
    callback = [SaveModelCallback(monitor=monitor,comp=np.greater,fname=model_fname),TensorBoardCallback(log_dir=os.path.join( "/home/wangjingqi/hthb/log/tensorboard",model_fname), projector=False,trace_model=False,log_preds=False)] 
    learn.fit_one_cycle(config.full_epoch, lr_max=config.full_lr_max,
        cbs=callback,)
    if isinstance(learn.model,nn.DataParallel):
        learn.model = learn.model.module
    saved = dict(model = learn.model.state_dict(),image_size=config.image_size)
 
    torch.save(saved,os.path.join(config.ck,model_fname+".pth"))
    
    for i in learn.recorder.values:
        log.loc[len(log)] =[round(j, 3) for j in i]
    log_name = model_fname.split("_")[-1]
    log.to_csv(os.path.join(config.log,f"{config.image_size}_{config.model_name}_{log_name}-{fold}-train.csv"))
    pickle.dump(config.count,open(f"/home/wangjingqi/hthb/th/{config.model_name}_{log_name}-{fold}-th.pkl", 'wb'))
    


