# %%

import os
import cv2

import numpy as np
import pandas as pd
from PIL import Image
import tifffile as tiff
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from config import config
from fda import transfer

# %%
OUT_TRAIN =config.TRAIN[0]
OUT_MASKS = config.MASKS[0]
os.makedirs(OUT_TRAIN, exist_ok=False)
os.makedirs(OUT_MASKS, exist_ok=False)

#the size of tiles


MASKS = config.LABELS
DATA = config.Data

# %%
# functions to convert encoding to mask and mask to encoding


def enc2mask(mask_rle, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs
df_masks = pd.read_csv(MASKS)[['id', 'organ', 'rle']].set_index('id')#add



class HuBMAPDataset(Dataset):
    def __init__(self, idx, encs=None):
        self.image = cv2.imread(os.path.join(DATA,str(idx)+'.tiff'),)#bgr
        self.shape = self.image.shape
        self.mask = enc2mask(encs,(self.shape[1],self.shape[0])) if encs is not None else None
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):

        image = cv2.resize(self.image,(config.image_size,config.image_size),
                            interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(self.mask,(config.image_size,config.image_size),
                             interpolation = cv2.INTER_LINEAR)

        return image, mask,idx

x_tot,x2_tot = [],[]
idxs = []
mask_ = []
mask_map = config.organ2label
with tqdm(desc='image-processing', unit='it', total=len(df_masks)) as pbar:
    for index,(organ, encs) in df_masks.iterrows():
        #image+mask dataset
        ds = HuBMAPDataset(index,encs=encs)
        for i in range(len(ds)):
            im,m,idx = ds[i]
            x_tot.append((im/255.0).reshape(-1,3).mean(0))
            x2_tot.append(((im/255.0)**2).reshape(-1,3).mean(0))
            if config.num_classes>2:
                m = mask_map[organ]*m 
            for e in np.unique(m): #add
                mask_.append(e) #add
            #write data   
         
            cv2.imwrite(os.path.join( OUT_TRAIN,f'{index}.png'),im)#保存是rgb
            Image.fromarray(m).save(os.path.join( OUT_MASKS,f'{index}.png'))
        pbar.update()

#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', img_std)
print('classes of mask', set(mask_)) 

# %%


# %%
print(len(os.listdir(OUT_TRAIN)),len(os.listdir(OUT_MASKS)))


