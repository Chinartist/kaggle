import torch
import torch.nn as nn
import torch.nn.functional as F
class model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10),nn.Dropout(0.5))
    def update_dropout(self, dropout):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
m = model()
m.update_dropout(0.2)
print(m)
m.update_dropout(0.1)
print(m)