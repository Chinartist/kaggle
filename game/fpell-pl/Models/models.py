
import torch 
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel,AutoTokenizer
import pytorch_lightning as pl
class MaxPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,hidden_state,attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        hidden_state[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(hidden_state, 1)[0]
        return max_embeddings

class MeanPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,hidden_state, attention_mask):

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_state.size())
        )
        mean_embeddings = torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
        return mean_embeddings

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start, layer_weights = None):
        super().__init__()
        if layer_start < 0:
            self.layer_start =num_hidden_layers+1+layer_start
        else:
            self.layer_start = layer_start
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - self.layer_start), dtype=torch.float)
            )

    def forward(self, features):

        all_layer_embedding = torch.stack(features)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weighted_average
