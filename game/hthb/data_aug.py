from pickletools import uint8
import random
from albumentations import *
from config import config
import torch
from torch.nn import functional as F
import numpy as np
import cv2
def img_add(img_src, img_main, mask_src):

    mask = np.asarray(mask_src, dtype=np.uint8)
    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask*255)
    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),mask = mask*255)
    img_main = img_main - sub_img02 + sub_img01

    return img_main

def muti_copy_paste(mask_src, img_src,mask, img):
    for i in range(len(mask_src)):
        img = img_add(img_src[i], img, mask_src[i])
        mask = img_add(mask_src[i], mask, mask_src[i])
    return img,mask
def copy_paste(mask_src, img_src, mask_main, img_main):
    img = img_add(img_src, img_main, mask_src)
    mask = img_add(mask_src, mask_main, mask_src)
    return img,mask
def dropout(img,mask,p=0.3):
    m = torch.ones(config.image_size,config.image_size,1)
    m = F.dropout(m,p)
    m = np.asarray(m,dtype=np.uint8)
    ms = np.concatenate([m,m,m],axis=-1)
    img = img*ms
    mask = mask*m.squeeze(-1)
    return img,mask

def selfcopypaste(img,mask,cropsize=int(config.image_size//2),fold=0):
    src_img = np.zeros(img.shape,dtype=np.uint8)
    src_mask = np.zeros(mask.shape,dtype=np.uint8)
    for x in range(mask.shape[0]):
        if mask[x].sum() >0:
            break
    for y in range(mask.shape[1]):
        if mask[:,y].sum() >0:
            break
    x1 = x
    x2 = x + cropsize if x + cropsize  <= config.image_size -1 else config.image_size-1
    y1 = y
    y2 = y + cropsize if y + cropsize  <= config.image_size -1 else config.image_size-1

    x_start = int(config.image_size//2-0.5*(x2-x1))
    y_start = int(config.image_size//2-0.5*(y2-y1))
    src_img[x_start:x_start+x2-x1,y_start:y_start+y2-y1] = img[x1:x2,y1:y2]
    src_mask[x_start:x_start+x2-x1,y_start:y_start+y2-y1] = mask[x1:x2,y1:y2]
    transfered = transfer[fold](image=src_img,mask=src_mask)
    src_img,src_mask = transfered["image"],transfered["mask"]
    img,mask = copy_paste(img_main=img,img_src=src_img,mask_main=mask,mask_src=src_mask)

    return img,mask



    
def CutOut(img,mask):
    m = np.ones((config.image_size,config.image_size,1),dtype=np.uint8)
    transfer = Compose(
        [
            CoarseDropout(max_holes = 64,min_holes=32,max_height=64,max_width=64,min_height=16,min_width=16,fill_value=0,p=1)
        ]
    )
    m = transfer(image=m)["image"]
    ms = np.concatenate([m,m,m],axis=-1)
    img = img*ms+(1-ms)*245
    mask = mask*m.squeeze(-1)
    return img,mask


def get_p(p = [0.3,0.35,0.4,0.45,0.5,0.55]):
    return p[random.randint(0,len(p)-1)]


transfer = [
    Compose([Transpose(1),],p=1),
    Compose([Transpose(1),],p=1),
    Compose([Transpose(1),],p=1),
    Compose([Transpose(1),],p=1),
    Compose([Transpose(1),],p=1),
    # Compose([HorizontalFlip(1),],p=1),
    # Compose([VerticalFlip(1),],p=1),
    # Compose([RandomRotate90(1),],p=1),
    # Compose([RandomRotate90(1),Transpose(1)],p=1)


]



def get_aug(p=1):
    return Compose([
    OneOf(
        [
    HorizontalFlip(p=get_p()),
    VerticalFlip(p=get_p()),
    RandomRotate90(p=get_p()),
        ],p=get_p()
    ),
    Transpose(p=get_p()),

    RandomSizedCrop(min_max_height=(int(config.image_size*0.7),config.image_size), height=config.image_size, width=config.image_size, p=get_p()),
    # Morphology
    ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.2), rotate_limit=(-30, 30),
                     interpolation=1, border_mode=0, value=(0, 0, 0), p=get_p()),
    GaussNoise(var_limit=(0, 50.0), mean=0, p=get_p()),
    GaussianBlur(blur_limit=(3, 7), p=get_p()),
    # Color
    RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5,
                             brightness_by_max=True, p=get_p()),
    HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
                       val_shift_limit=0, p=get_p()),
    OneOf([
        OpticalDistortion(p=get_p([0.1,0.2,0.3])),
        GridDistortion(p=get_p([0.1,0.2,0.3])),
        IAAPiecewiseAffine(p=get_p([0.1,0.2,0.3])),
    ], p=get_p([0.1,0.2,0.3])),
], p=p)
def get_valaug(p=0.4):
   return Compose([
    
    RandomRotate90(p=0.2),

    RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5,
                             brightness_by_max=True, p=0.2),
    HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
                       val_shift_limit=0, p=0.3),
    
], p=p)
