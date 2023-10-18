# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from replknet import create_RepLKNet31B,create_RepLKNet31L,create_RepLKNetXL
# class Decoder(nn.Module):
#     def __init__(
#             self,
#             in_ch=[],
#             out_ch=None,
#     ):
#         super().__init__()
#         self.mlp = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(dim, out_ch, 1, padding=0, bias=False),  # follow mmseg to use conv-bn-relu
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Flatten(-3)
#             ) for i, dim in enumerate(in_ch)])
#         self.fuse = nn.Sequential(
#             nn.Linear(len(in_ch) * out_ch, out_ch,bias=False),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, feature):
#         out = []
#         for i, f in enumerate(feature):
#             f = self.mlp[i](f)
#             out.append(f)
#         x = self.fuse(torch.cat(out, dim=1))
#         return x

# class Model(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         # model = timm.create_model(config.model_name, pretrained=config.pretrained, in_chans=2,drop_rate=0.3, drop_path_rate=0.3)
#         # clsf = model.default_cfg['classifier']
#         # n_features = model._modules[clsf].in_features
#         # model._modules[clsf] = nn.Identity()
        
#         if "RepLKNet" in config.model_name:
#             encoder = create_RepLKNet31B()
            
#             if config.pretrained:
#                 print("loading pretrained model")
#                 encoder_param = encoder.state_dict()
#                 pretrained = torch.load(f"/home/wangjingqi/input/ck/g2net/pretrained/{config.model_name}.pth", map_location="cpu")
#                 pretrained_param = {}
#                 for k,v in pretrained.items():
#                     if k in encoder_param:
#                         if v.shape != encoder_param[k].shape:
#                             print('pop', k, v.shape, encoder_param[k].shape)
#                         else:
#                             pretrained_param[k] = v
#                     else:
#                         print('pop', k, v.shape)
#                 encoder.load_state_dict(pretrained_param, strict=False)
#             n_features = 128
#             self.encoder = encoder
#             self.decoder = Decoder([128,256,512,1024], n_features)
#             self.head = nn.Sequential(nn.Linear(n_features, n_features),  nn.ReLU(),nn.Dropout(0.3), nn.Linear(n_features, 1))
#         else:
#             encoder = timm.create_model(config.model_name, pretrained=config.pretrained, in_chans=2,drop_rate=0.3, drop_path_rate=0.3)
#             clsf = encoder.default_cfg['classifier']
#             n_features = encoder._modules[clsf].in_features
#             encoder._modules[clsf] = nn.Identity()
#             self.encoder = encoder
#             self.head = nn.Sequential(nn.Linear(n_features, n_features),  nn.ReLU(),nn.Dropout(0.3), nn.Linear(n_features, 1))
        
#     def forward(self, x):
#         x = self.encoder(x)
#         if isinstance(x, list):
#             x = self.decoder(x)
#         x = self.head(x)
#         return x.squeeze(-1)

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