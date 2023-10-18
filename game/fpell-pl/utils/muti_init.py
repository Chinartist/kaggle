from torch import nn
def xavier_uniform_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                            m.bias.data.zero_()
def xavier_normal_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                            m.bias.data.zero_()
def he_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in',nonlinearity='relu')
            if m.bias is not None:
                            m.bias.data.zero_()
def kiming_init(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                            m.bias.data.zero_()
def orthogonal_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=1)
            if m.bias is not None:
                            m.bias.data.zero_()