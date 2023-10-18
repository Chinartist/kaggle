import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from replknet import create_RepLKNet31B,create_RepLKNet31L,create_RepLKNetXL
import torchaudio
class conv_stem(nn.Module):
    def __init__(self,conv_stem) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 32, kernel_size=(31, 63), stride=(1, 1), padding=(15, 31), bias=False)
        self.pool = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
        self.conv_stem = conv_stem
        self.time_mask_num = 1 # number of time masking
        self.freq_mask_num = 1 # number of frequency masking
        self.transforms_time_mask = nn.Sequential(
                torchaudio.transforms.TimeMasking(time_mask_param=10),
            )

        self.transforms_freq_mask = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
            )
    def forward(self,x):
        x = self.conv(x)
        x = self.pool(x)
        if self.training:
            for _ in range(self.time_mask_num): # tima masking
                x = self.transforms_time_mask(x)
            for _ in range(self.freq_mask_num): # frequency masking
                x = self.transforms_freq_mask(x)
        x = self.conv_stem(x)
        return x
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.shape[0] in [1024]:
            in_chans = 32
        else:
            in_chans = 2
        encoder = timm.create_model(config.model_name, pretrained=config.pretrained, in_chans=in_chans,drop_rate=config.drop)
        if config.shape[0] in [1024]:
            encoder.conv_stem = conv_stem(encoder.conv_stem)
        clsf = encoder.default_cfg['classifier']
        n_features = encoder._modules[clsf].in_features
        encoder._modules[clsf] = nn.Identity()
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.Linear(n_features,1, bias=True))
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x.squeeze(-1)
def splitter(m):
    if isinstance(m, nn.DataParallel):
        m = m.module
    group0 = []
    group1 = []
    for n,p in m.named_parameters():
        if 'conv_stem' in n or 'classifier' in n:
            group0.append(p)
        else:
            group1.append(p)
    return [group0,group1]
# from collections import deque
# class conv_bn_silu_block(nn.Module):
#     def __init__(self, in_ch, out_ch, k):
#         super().__init__()
#         self.block=nn.Sequential(
#             nn.Conv1d(in_ch, out_ch, k),
#             nn.BatchNorm1d(out_ch),
#             nn.SiLU()
#         )
        
#     def forward(self, x):
#         return self.block(x)
    
# class Model(nn.Module):
#     def __init__(self, config,in_ch=4096, out_ch=360, k=12):
#         super().__init__()
#         self.save_models = deque(maxlen=3)
#         self.c1 = conv_bn_silu_block(in_ch, out_ch, k)
#         self.c2 = conv_bn_silu_block(in_ch, out_ch, k * 2)
#         self.c3 = conv_bn_silu_block(in_ch, out_ch, k // 2)
#         # self.c4 = conv_bn_silu_block(in_ch, out_ch, k // 4)
#         # self.c5 = conv_bn_silu_block(in_ch, out_ch, k * 4)
#         self.c6 = conv_bn_silu_block(out_ch, 1, 1)  # in_ch * 5 + out_ch 
#         self.fc = nn.Linear(1041, 1)  #1712  
    
    
#     def forward(self, x):
#         # , self.c4(x), self.c5(x)
#         x = torch.cat([self.c1(x), self.c2(x), self.c3(x)], dim=2)
#         x = self.c6(x)
#         x = self.fc(x.view(-1,1041))
#         return x