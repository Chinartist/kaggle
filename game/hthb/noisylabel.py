# %%
from config import config
from fastai.vision.all import *
import pandas  as pd
import cv2
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import KFold

# %%
mean = np.array(config.mean )
std = np.array(config.std)
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

os.makedirs(config.MASKS[1],exist_ok=True)
seed_everything(config.seed)
th=0.5
score_th = 0.5
device = 4
log = pd.DataFrame(columns=["id","organ","score"])
def dice(pred,targ):
    pred,targ = flatten_check(pred, targ)
    pred = (pred>th).float()
    inter = (pred*targ).float().sum().item()
    union = (pred+targ).float().sum().item()
    return 2.0 * inter/union if union > 0 else None

# %%
def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class HuBMAPDataset(Dataset):
    def __init__(self,fold):
        ids = pd.read_csv(config.LABELS,).id.astype(str).values
        organs = pd.read_csv(config.LABELS,).organ.astype(str).values
        self.id2organ = dict(zip(ids,organs))
        kf = KFold(n_splits=config.nfolds,random_state=config.seed,shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][1]])
        self.fnames = [fname for fname in os.listdir(config.TRAIN[0]) if fname.split('.')[0] in ids]
     
    def __len__(self):
        return len(self.fnames)
    def get_image(self, fname,i):
      
        img = cv2.cvtColor(cv2.imread(os.path.join(config.TRAIN[i],fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(config.MASKS[0],fname),cv2.IMREAD_GRAYSCALE)
        return img,mask

    def __getitem__(self, idx):
        fname = self.fnames[idx]

        img, mask = self.get_image(fname,0)
        return img2tensor((img/255.0 - mean)/std),self.id2organ[fname.split('.')[0]],img2tensor(mask),fname.split(".")[0]


# %%
def t(model):
    ck = {}
    for k,v in model.items():
        if 'cls' in k.split(".") and "weight" in k.split("."):
            k = "logit.weight"
            ck.update({k:v})
        elif 'cls' in k.split(".") and "bias" in k.split("."):
                k = "logit.bias"
                ck.update({k:v})
        else:
            ck.update({k:v})
    return ck

# %%
checkpoint_path = '/home/wangjingqi/hthb/k-fold-checkpoint'
# selected_model = ["-cutout2021"]
selected_model = ["nodropoutcutout2021"]\
+["--label-nosample2021","all-label-nosample2021","label2021","-label2021","--label2021","---label2021"]
MODELS =[ os.path.join(checkpoint_path,model_name) for model_name in os.listdir(checkpoint_path) if  model_name.split("_")[-1].split(".")[0] in selected_model ]
models = []

for fold in range(config.nfolds):
    ps = []
    m = []
    for path in MODELS:
        if path.split("_")[-2] == str(fold):
            ps.append(path)
    print(ps)
    for p in ps:
        state_dict = torch.load(p,map_location=torch.device('cpu'))
        state_dict = t(state_dict)
        if os.path.basename(path).split("_")[0] == "1" :
            model,_ = config.models[config.model_name.split("-")[0]]
            model = model(config=config,pretrain=False)
        model.load_state_dict(state_dict= state_dict)
        model.float()
        model.eval()
        model.to(device)
        m.append(model)
    models.append(m)

del state_dict



# %%
print(len(models[0]))

# %%
from PIL import Image

# %%
def model_pred(models,img,mask):
    preds = 0
    for model in models:
        pred = model(img,None)[0]
        preds +=pred
    flips = [[-1],[-2],[-2,-1]]
    for f in flips:
        xf = torch.flip(img,f)
        for model in models:
            pred = model(xf,None)[0]
            pred = torch.flip(pred,f)
            preds += pred
    preds /= (1+len(flips)) 
    preds /= len(models)
    score = dice(torch.sigmoid(preds),mask)

    return score,torch.sigmoid(preds).squeeze(0).squeeze(0).cpu()>th
# def model_pred(models,img,mask):
#     preds = 0
#     for model in models:
#         pred = model(img,None)[0]
#         preds +=pred

#     preds /= len(models)
#     score = dice(torch.sigmoid(preds),mask)

#     return score,torch.sigmoid(preds).squeeze(0).squeeze(0).cpu()>th

# %%
with torch.no_grad():
    for fold in range(config.nfolds):
        if len(models[fold]) == 0:
            break
        data = HuBMAPDataset(fold=fold)
        with tqdm(desc=f"test-{fold}", unit='it', total=len(data)) as pbar:
            for i in range(len(data)):
                img,o,mask,idx = data[i]
                img = img.to(device)
                mask = mask.to(device)
                score,preds = model_pred(models[fold],img.unsqueeze(0),mask.unsqueeze(0))
                if score < score_th :
                    p_mask = Image.fromarray(np.asarray(preds,dtype=np.uint8))
                    p_mask.save(os.path.join(config.MASKS[1],f"{idx}.png"))
                    src_mask = cv2.resize(cv2.imread(os.path.join(config.MASKS[1],f"{idx}.png"),cv2.IMREAD_GRAYSCALE),(1024,1024),interpolation = cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join ("/home/wangjingqi/input/hubmap-organ-segmentation/1024/1/tta_all_masks/",f"{idx}.png"),src_mask)
                    src_mask = cv2.resize(cv2.imread(os.path.join(config.MASKS[1],f"{idx}.png"),cv2.IMREAD_GRAYSCALE),(1536,1536),interpolation = cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join ("/home/wangjingqi/input/hubmap-organ-segmentation/1536/1/tta_all_masks/",f"{idx}.png"),src_mask)
                    log.loc[len(log)] = [idx,o,score]
                pbar.update()
    

# %%
log.to_csv("/home/wangjingqi/hthb/log/test.csv") 

# %%
import pandas as pd

log_ = pd.read_csv("/home/wangjingqi/hthb/log/test.csv")[["id","organ","score"]]

# %%
log_["score"].plot()

# %%
log_#.describe()

# %%
select = log_[log_["score"] < score_th ]
select.to_csv("/home/wangjingqi/hthb/log/select.csv")
select = pd.read_csv("/home/wangjingqi/hthb/log/select.csv")
select["score"].plot()

# %%
def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


# %%
i=0

# %%
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from config import config
select = pd.read_csv("/home/wangjingqi/hthb/log/select.csv")
i +=1
idx =select.loc[i]["id"]
mask = np.asarray (Image.open(os.path.join(config.MASKS[0],f"{idx}.png")))
p_mask = np.asarray (Image.open(os.path.join(config.MASKS[1],f"{idx}.png")))
img = cv2.imread(os.path.join(config.TRAIN[0],f"{idx}.png"))
visualize(img,p_mask,img,mask)

# %%
select


