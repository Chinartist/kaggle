from types import SimpleNamespace
from muti_init import xavier_uniform_init,xavier_normal_init,he_init,kiming_init,orthogonal_init
from segformer import segformer,segformer_layers
from Fsegformer import Fsegformer,Fsegformer_layers
from Usegformer import Usegformer,Usegformer_layers
from senformer import senformer,senformer_layers
from HorNet import hornet,hornet_layers
from coat import coat,coat_layers
from ecoat import ecoat,ecoat_layers
from scoat import scoat,scoat_layers
from fcoat import fcoat,fcoat_layers
import os
from collections import defaultdict

'''

muti scale


'''
'''
to do...



mutisampledropoutblock
'''

'''
"label2021","-label2021","--label2021"
th=0.55
score_th = 0.45
--label

'''

config = SimpleNamespace(**{})
config.folds = [0,1,2,3,4]
# config.folds = [0,1,2,3,4]

config.init = [xavier_uniform_init,xavier_normal_init,he_init,kiming_init,orthogonal_init]

config.seed = 2021
config.NUM_WORKERS = 16
config.device_ids =[[3,2,1,0],[4,5,6,7],[0,1,2,3,4,5,6,7]][1]
config.mean = [0.485, 0.456, 0.406]
config.std = [0.229, 0.224, 0.225]
config.img_scale = [(512,512),(768,768),(896,896),(1024,1024)]
config.count ={0: defaultdict(int),1: defaultdict(int),2: defaultdict(int),3: defaultdict(int),4: defaultdict(int)}

config.num_classes = 1
config.add_extra_convs = "on_lateral"
config.num_outs = 5
config.image_size = 1024
config.mit = 2
config.branch_depth = 6
config.nfolds = 5

config.bs = 10
config.head_epoch = 8
config.full_epoch = 200

config.head_lr_max =5e-3
config.full_lr_max =slice(2.5e-4,2.5e-3)

# config.model_name = f"segformer-b{config.mit}"
# config.model_name = f"Fsegformer-b{config.mit}"
# config.model_name = f"hornet-b{config.mit}"
# config.model_name = f"senformer-b{config.mit}"
# config.model_name = f"Usegformer-b{config.mit}"
config.model_name = f"coat"
# config.model_name = f"fcoat"
# config.model_name = f"ecoat"
# config.model_name = f"scoat"
config.organ2label =dict(
    kidney=1,
    prostate=2,
    largeintestine=3,
    spleen=4,
    lung=5)#add


config.log = "/home/wangjingqi/hthb/log"
config.ck = '/home/wangjingqi/hthb/submit/'
config.pretrain = "/home/wangjingqi/hthb/models"
config.MASKS = [f'/home/wangjingqi/input/hubmap-organ-segmentation/{config.image_size}/{config.num_classes}/masks',
f'/home/wangjingqi/input/hubmap-organ-segmentation/{config.image_size}/{config.num_classes}/tta_lung_masks']

config.TRAIN = [f'/home/wangjingqi/input/hubmap-organ-segmentation/{config.image_size}/{config.num_classes}/hpa_images',
f'/home/wangjingqi/input/hubmap-organ-segmentation/{config.image_size}/{config.num_classes}/hubmap_images']
config.LABELS = "/home/wangjingqi/input/hubmap-organ-segmentation/train.csv"
config.Data = '/home/wangjingqi/input/hubmap-organ-segmentation/train_images'

config.models = {
                "senformer":[senformer,senformer_layers],
                "segformer":[segformer,segformer_layers],
                "Fsegformer":[Fsegformer,Fsegformer_layers],
                "hornet":[hornet,hornet_layers],
                "Usegformer":[Usegformer,Usegformer_layers],
                "coat":[coat,coat_layers],
                "scoat":[scoat,scoat_layers],
                "ecoat":[ecoat,ecoat_layers],
                "fcoat":[fcoat,fcoat_layers]

                }
