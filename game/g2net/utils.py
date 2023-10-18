from fastai.vision.all import *
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

#create Metrics
class roc(Metric):#
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): 
        self.y = None
        self.pred = None
    def accumulate(self, learn):
        y =learn.y
        y = (y>0).float()
        pred = learn.pred.sigmoid()
        if self.y is None: self.y = y
        else: self.y = torch.cat((self.y, y))
        if self.pred is None: self.pred = pred
        else: self.pred = torch.cat((self.pred, pred))
    @property
    def value(self): 
        try:
            roc = roc_auc_score(self.y.cpu().numpy(), self.pred.cpu().numpy())
            return roc
        except:
            return -1
class acc(Metric):#
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): 
        self.y = None
        self.pred = None
    def accumulate(self, learn):
        y =learn.y
        y = (y>0).float()
        pred = (learn.pred.sigmoid()>=0.5).float()
        if self.y is None: self.y = y
        else: self.y = torch.cat((self.y, y))
        if self.pred is None: self.pred = pred
        else: self.pred = torch.cat((self.pred, pred))
    @property
    def value(self): 
        try:
            acc = accuracy_score(self.y.cpu().numpy(), self.pred.cpu().numpy())
            return acc
        except:
            return -1
