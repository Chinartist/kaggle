import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def load_and_resave(dir):
    data_name = os.listdir(dir)
    data_name = [i for i in data_name if i.endswith(".hdf5")]
    for i in tqdm(range(len(data_name))):
        data_path = os.path.join(dir, data_name[i])
        with h5py.File(data_path, 'r') as f:
            g = f[data_name[i].split('.')[0]]
            h1 = g["H1"]['SFTs'].__array__()
            l1 = g["L1"]['SFTs'].__array__()
            data= {'H1': h1, 'L1': l1}
        np.save(data_path.split('.')[0]+".npy", data)
        os.remove(data_path)

if __name__ == "__main__":
    load_and_resave("/home/wangjingqi/input/dataset/g2net/test")
