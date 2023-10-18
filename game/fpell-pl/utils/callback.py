from fastai.text.all import *

class TkCallback(Callback):
    "A `Callback` that keeps track of the best value in `monitor`."
    order,remove_on_fetch,_only_train_loop = 60,True,True
    def __init__(self, 
        monitor='valid_loss', # value (usually loss or metric) being monitored.
        comp=None, # numpy comparison operator; np.less if monitor is loss, np.greater if monitor is metric.
        min_delta=0., # minimum delta between the last monitor value and the best monitor value.
        reset_on_fit=True ,# before model fitting, reset value being monitored to -infinity (if monitor is metric) or +infinity (if monitor is loss).
        best = None
    ):
        if comp is None: comp = np.less if 'loss' in monitor or 'error' in monitor else np.greater
        if comp == np.less: min_delta *= -1
        self.monitor,self.comp,self.min_delta,self.reset_on_fit,self.best= monitor,comp,min_delta,reset_on_fit,best

    def before_fit(self):
        "Prepare the monitored value"
        self.run = not hasattr(self, "lr_finder") and not hasattr(self, "gather_preds")
        if self.reset_on_fit or self.best is None: self.best = float('inf') if self.comp == np.less else -float('inf')
        assert self.monitor in self.recorder.metric_names[1:]
        self.idx = list(self.recorder.metric_names[1:]).index(self.monitor)

    def after_epoch(self):
        "Compare the last value to the best up to now"
        val = self.recorder.values[-1][self.idx]
        if self.comp(val - self.min_delta, self.best): self.best,self.new_best = val,True
        else: self.new_best = False

    def after_fit(self): self.run=True

class FGMCallback(Callback):

    order=TrackerCallback.order+2
    def __init__(self, 
        model
    ):
        self.fgm =FGM(model)
    def after_backward(self,):
        self.fgm.attack()
        pred = self.learn.model(*self.xb)
        loss_grad = self.loss_func(pred, *self.yb)
        loss_grad.backward()
        self.fgm.restore()



class FGM():
    def __init__(self, model):
        if isinstance(model,nn.DataParallel):
            model = model.module
        self.model = model
        self.backup = {}
        self.emb_name = 'encoder.embeddings.word_embeddings.weight'
    def attack(self, epsilon=1.):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
