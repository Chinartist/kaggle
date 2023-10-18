# %%
from notebook.config import config
import torch
import os
def config_model():
    config.ck = os.path.join(config.ck,config.model_fname)
    cks = os.listdir(config.ck)
    for ck in cks:
        print(ck)
        ck_path = os.path.join(config.ck,ck)
        checkpoint = torch.load(ck_path,map_location="cpu")["state_dict"]
        torch.save(dict(model=checkpoint,config=config),ck_path)
      
        





# %%
config_model()


