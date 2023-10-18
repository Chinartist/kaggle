
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32


from timm.models.vision_transformer import Attention#, Mlp
from timm.models.layers import DropPath
from einops import rearrange


class Mlp(nn.Module):
    """ Multilayer perceptron from timm library."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    '''
    Taken from timm library Attention module
    with slight modifications to do Cross-Attention.
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
    def forward(self,q_in, kv_in):
        B, N, C = kv_in.shape
        _, L, _ = q_in.shape
        # Create key and value tokens
        kv = self.kv(kv_in).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # Create query tokens
        q = self.to_q(q_in)
        q = q.reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#--- Decoder Block ---
#         │
#         ├────────┐
# ┌───────┴───────┐│
# │      MLP      ││
# └───────┬───────┘│
#         ├────────┘
#         ├────────┐
# ┌───────┴───────┐│
# │Self-Attention ││
# └───────┬───────┘│
#         ├────────┘
#         ├────────┐
# ┌───────┴───────┐│
# │Cross-Attention││
# └───────┬───────┘│
#         ├────────┘
#         │
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads,mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=0.)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            attn_drop=attn_drop, proj_drop=0.)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)

    def forward_crossattn(self, queries, features):
        ##--------- Cross-Attention Block
        out = queries + self.drop_path1(self.cross_attn(self.norm1(queries), features))
        return out

    def forward_attn(self, q):
        ##--------- Self-Attention Block
        q = q + self.drop_path2(self.attn(self.norm2(q)))
        return q

    def forward_mlp(self, q):
        ##--------- MLP Block
        cls_features = q + self.drop_path3(self.mlp(self.norm3(q)))
        return cls_features

    def forward(self, queries, features):
        out = self.forward_crossattn(queries, features)
        out = self.forward_attn(out)
        out = self.forward_mlp(out)
        return out

class TransformerLearner(nn.Module):
    def __init__(self, dim, num_heads, num_queries, branch_depth, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TransformerLearner, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, branch_depth)]
        self.layers = nn.ModuleList([
            DecoderBlock(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                 attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
            for i in range(branch_depth)
        ])
        # Learnable class embeddings
        self.queries = nn.Parameter(torch.zeros(1, num_queries, dim), requires_grad=True)
        # Norm features and class embeddings for training stability
        self.norm_features = nn.LayerNorm(dim)
        self.norm_embs = nn.LayerNorm(dim)

    def forward(self, features):
        B, _, H, W = features.shape
        # Tokenize 2D feature maps
        features = rearrange(features, 'b c h w -> b (h w) c')
        features = self.norm_features(features)

        # Expand to batch size
        cls_embs = self.queries.expand(B, -1, -1)

        # Decoder
        for layer in self.layers:
            cls_embs = layer(cls_embs, features)

        # Norm class embeddings for stability
        cls_embs = self.norm_embs(cls_embs)
        # Prediction
        pred = (features @ cls_embs.transpose(-2, -1))#bs,hw,dim *bs,dim, nq->bs,hw,nq
        # Reshape into 2D maps
        pred = rearrange(pred, 'b (h w) c -> b c h w', h=H, w=W)

        return pred