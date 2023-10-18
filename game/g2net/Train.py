from fastai.vision.all import *
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold,KFold
from G2NetDataset import G2NetDataset
from utils import  roc, acc
from Model import Model,splitter
from transformers import PretrainedConfig
I = 0
class Config(PretrainedConfig):

    def __init__(self,
    
    model_name="inception_v4",
    save_name ="inception_v4",
    bs=32,lr=(4e-4,4e-4),wd=1e-6,
    epochs = 25,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    seed = 42,
    nfolds = 5,
    num_workers = 8,
    device_ids = [0],
    used_folds = [0,1,2,3,4],
    weight = 1.0,
    pretrained = True,
    use_generated_data = False,
    gfolds = 10,
    shape=[128,32],
    drop = 0.,
    transformers_version=None
    ):
        self.save_name = save_name 
        self.model_name = model_name
        self.bs = bs
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.seed = seed
        self.nfolds = nfolds
        self.num_workers = num_workers
        self.device_ids = device_ids
        self.used_folds = used_folds
        self.weight = weight
        self.pretrained = pretrained
        self.use_generated_data = use_generated_data
        self.gfolds = gfolds
        self.shape = shape
        self.drop = drop
def Loss(pred, target):
    b_target = (target>0).float()
    return nn.BCEWithLogitsLoss()(pred, b_target)#+nn.SmoothL1Loss()(pred[:,1], target)
if __name__ == "__main__":
    config = Config()
    print(config)
    pl.seed_everything(config.seed)
    os.makedirs(f"/home/wangjingqi/input/ck/g2net/{config.save_name}/log", exist_ok=True)
    config.to_json_file(f"/home/wangjingqi/input/ck/g2net/{config.save_name}/config.json")

    kfold = KFold(n_splits=config.nfolds, random_state=config.seed, shuffle=True)
    id2label = pd.read_csv("/home/wangjingqi/input/dataset/g2net/generted_train/generted_train_labels.csv")
    id2label = id2label[id2label.target >= 0].reset_index(drop=True).head(10000)
    for nfold, (train_idx, val_idx) in enumerate(kfold.split(id2label.id.values, id2label.target.values)):
        id2label.loc[val_idx, 'fold'] = int(nfold)
    if config.use_generated_data:
        generated_data = pd.read_csv("/home/wangjingqi/input/dataset/g2net/generted_train/generted_train_labels.csv")
        kfold = KFold(n_splits=config.gfolds, random_state=config.seed, shuffle=True)
        for nfold, (train_idx, val_idx) in enumerate(kfold.split(generated_data.id.values, generated_data.target.values)):
            generated_data.loc[val_idx, 'fold'] = int(nfold)
    for fold in config.used_folds:
        model = Model(config)

        print(model.classifier)
        train_df = id2label[id2label.fold != fold].reset_index(drop=True)
        if config.use_generated_data:
            generated_train_df = generated_data[generated_data.fold.isin(random.sample(list(np.arange(config.gfolds)),5)) ].reset_index(drop=True)
            train_df = pd.concat([train_df,generated_train_df],axis=0).reset_index(drop=True)
        # num_positives = torch.sum(torch.Tensor(train_df.target), dim=0)
        # num_negatives = len(train_df.target) - num_positives
        # pos_weight  = num_negatives / num_positives
        # print(f"正样本数：{num_positives}，负样本数：{num_negatives}，pos_weight：{pos_weight}")
        
        valid_df = id2label[id2label.fold == fold].reset_index(drop=True)
        ds_t = G2NetDataset(config,dir =["/home/wangjingqi/input/dataset/g2net/train","/home/wangjingqi/input/dataset/g2net/generted_train" ],df=train_df,train=True)
        ds_v = G2NetDataset(config,dir=["/home/wangjingqi/input/dataset/g2net/train","/home/wangjingqi/input/dataset/g2net/generted_train"],df = valid_df,train=False)

        print("样本数：",len(ds_t),len(ds_v))

        data = DataLoaders.from_dsets(ds_t,ds_v,bs=config.bs,num_workers=config.num_workers,pin_memory=True).to(config.device_ids[0])

        
        if len(config.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=config.device_ids)
        model.to(config.device_ids[0])  
        metrics = [roc,acc]
        monitor = "roc"
        comp=np.greater
        learn = Learner(data, model,wd=config.wd ,lr = config.lr,loss_func=Loss,model_dir="",splitter=splitter,
                    metrics =metrics ,path=f"/home/wangjingqi/input/ck/g2net/{config.save_name}",cbs=[SaveModelCallback(monitor=monitor,comp=comp,fname=config.save_name+f"_{fold}"),CSVLogger(fname=f"log/{config.save_name}"+f"_{fold}.csv")]).to_fp16()
                
        #start with training the head
        print(f"Fold {fold}: {config.save_name}")
        learn.fit_one_cycle(config.epochs )
        



   