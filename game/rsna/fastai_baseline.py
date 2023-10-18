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

import albumentations as A
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import timm
import warnings
from nextvit import NextVitBNet,seresnext
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
def pfbeta(labels, predictions, beta=1.):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / max(y_true_count, 1)  # avoid / 0
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def optimal_f1(labels, predictions):
    thres = np.linspace(0, 1, 101)
    f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]
class pf1(Metric):
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): 
        self.y = None
        self.pred = None
    def accumulate(self, learn):
        y =learn.y
        pred = learn.pred.sigmoid()
        if self.y is None:
            self.y = y
            self.pred = pred
        else:
            self.y = torch.cat([self.y, y])
            self.pred = torch.cat([self.pred, pred])
    @property
    def value(self): 
        scores,th = optimal_f1(self.y.cpu().numpy(), self.pred.cpu().numpy())
        print(f"pf1: {scores} at threshold: {th}")
        return scores
    

class BreastCancerDataSet(torch.utils.data.Dataset):
    def __init__(self, df, cfg, transforms=None):
        super().__init__()
        self.df = df
        self.cfg = cfg
        self.transforms = transforms

    def __getitem__(self, i):

        path = self.cfg.image_dir+f'{self.df.iloc[i].patient_id}_{self.df.iloc[i].image_id}.png'
   
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        cancer_target = torch.as_tensor(self.df.iloc[i].cancer).float()
        return img, cancer_target
    def __len__(self):
        return len(self.df)
# %%
cfg = SimpleNamespace(**{})
cfg.img_size = (1024,768)#1536,960
cfg.pretrained=True
cfg.classes = ['cancer']
cfg.bs = 8
cfg.image_dir = "/home/wangjingqi/input/dataset/rsna/images/"

cfg.nfolds = 5
cfg.seed = 666
cfg.p = 0.5
cfg.device_ids = [1]
cfg.nw = 4
cfg.save_name = "NextVitBNet"
cfg.lr = 4e-5
cfg.wd = 0.0001
cfg.epochs = 10
print(cfg)
AUX_LOSS_WEIGHT = 0.
POSITIVE_TARGET_WEIGHT=20



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(cfg.seed)

# %%
train_aug = A.Compose([
    
    A.HorizontalFlip(p=cfg.p),
    A.RandomResizedCrop( height=cfg.img_size[0], width=cfg.img_size[1],scale=(0.8, 1),p=1),
], p=1)

val_aug = A.Compose([
    A.HorizontalFlip(p=cfg.p),
    A.RandomResizedCrop( height=cfg.img_size[0], width=cfg.img_size[1],scale=(0.8, 1), p=1),
], p=1)

from sklearn.model_selection import StratifiedGroupKFold

CATEGORY_AUX_TARGETS = ['site_id', 'laterality', 'view', 'implant', 'biopsy', 'invasive', 'BIRADS', 'density', 'difficult_negative_case', 'machine_id', 'age']
TARGET = 'cancer'
ALL_FEAT = [TARGET] + CATEGORY_AUX_TARGETS
df_train = pd.read_csv('/home/wangjingqi/input/dataset/rsna/train.csv')

kfold = StratifiedGroupKFold(n_splits = cfg.nfolds, shuffle = True, random_state = cfg.seed)
for num, (train_index, val_index) in enumerate(kfold.split(df_train, df_train['cancer'], df_train['patient_id'])):
    df_train.loc[val_index, 'fold'] = int(num)
df_train['fold'] = df_train['fold'].astype(int)
df_train.age.fillna(df_train.age.mean(), inplace=True)
df_train['age'] = pd.qcut(df_train.age, 10, labels=range(10), retbins=False).astype(int)
df_train[CATEGORY_AUX_TARGETS] = df_train[CATEGORY_AUX_TARGETS].apply(LabelEncoder().fit_transform)
AUX_TARGET_NCLASSES = df_train[CATEGORY_AUX_TARGETS].max() + 1


os.makedirs(f"/home/wangjingqi/input/ck/rsna/{cfg.save_name}/log", exist_ok=True)
def loss_fn(outputs, targets):
    preds = outputs
    cancer_target = targets
    cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        preds,
                        cancer_target,pos_weight=torch.tensor(POSITIVE_TARGET_WEIGHT).to(cfg.device_ids[0])
                    )
    loss = cancer_loss 
    return loss
# if __name__ == "__main__":

#         print(' ')

#         x_train = df_train
#         x_val = df_train
#         ds_t = BreastCancerDataSet(x_train,cfg,train_aug)
#         ds_v = BreastCancerDataSet(x_val,cfg,val_aug)
#         data = DataLoaders.from_dsets(ds_t,ds_v,bs=cfg.bs,
#                     num_workers=cfg.nw,pin_memory=True).to(cfg.device_ids[0])
#         model = NextVitBNet(AUX_TARGET_NCLASSES,path_dropout=0.2)
#         if len(cfg.device_ids) > 1:
#             model = torch.nn.DataParallel(model, device_ids=cfg.device_ids)
#         model.to(cfg.device_ids[0])
#         comp=np.greater
#         monitor = "pf1"
#         learn = Learner(data, model,wd=cfg.wd ,lr = cfg.lr,loss_func=loss_fn,model_dir="",metrics=[pf1],
#                     path=f"/home/wangjingqi/input/ck/rsna/{cfg.save_name}",cbs=[SaveModelCallback(monitor=monitor,comp=comp,fname=cfg.save_name,every_epoch=True),CSVLogger(fname=f"log/{cfg.save_name}"+f".csv")]).to_fp16()
#         print(f" {cfg.save_name}")
#         learn.fit_one_cycle(cfg.epochs )
if __name__ == "__main__":
    for fold in range(cfg.nfolds):
        print(' ')
        print(f"========== fold: {fold} training ==========")
        x_train = df_train[df_train['fold'] != fold]
        x_val = df_train[df_train['fold'] == fold]
        ds_t = BreastCancerDataSet(x_train,cfg,train_aug)
        ds_v = BreastCancerDataSet(x_val,cfg,val_aug)
        data = DataLoaders.from_dsets(ds_t,ds_v,bs=cfg.bs,
                    num_workers=cfg.nw,pin_memory=True).to(cfg.device_ids[0])
        model = NextVitBNet(path_dropout=0.2)
        # model =seresnext()

        if len(cfg.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=cfg.device_ids)
        model.to(cfg.device_ids[0])
        comp=np.greater
        monitor = "pf1"
        learn = Learner(data, model,wd=cfg.wd ,lr = cfg.lr,loss_func=loss_fn,model_dir="",metrics=[pf1],
                    path=f"/home/wangjingqi/input/ck/rsna/{cfg.save_name}",cbs=[SaveModelCallback(monitor=monitor,comp=comp,fname=cfg.save_name+f"_{fold}"),CSVLogger(fname=f"log/{cfg.save_name}"+f"_{fold}.csv")]).to_fp16()
        print(f"Fold {fold}: {cfg.save_name}")
        learn.fit_one_cycle(cfg.epochs )

# %%



