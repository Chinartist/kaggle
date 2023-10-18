import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import pyfstat
import random
from scipy import stats
import pandas as pd
import os
import shutil
import pandas as pd
def process( data: np.ndarray) -> np.ndarray:
        data = data* 1e22
        data = data.imag**2 + data.real**2
        data = data/data.mean()
        x = np.zeros((360, 4096))
        x[:, :data.shape[-1]] = data[:,:4096]
        return x
def clear_and_resave(path):
    frequency, timestamps, amplitudes = pyfstat.helper_functions.get_sft_as_arrays(
        path#writer.sftfilepath
    )
    path = path.split(".")[0].split("/")
    dir = "/".join(path[:-1])
    save_path = dir+".npy"
    h1 = process(amplitudes["H1"]).reshape(360,128,32).mean(axis=-1)
    l1 = process(amplitudes["L1"]).reshape(360,128,32).mean(axis=-1)
    sample = {"H1":h1,"L1":l1}

    np.save(save_path,sample)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    return None
def get_squared_snr(writer):
    snr = pyfstat.SignalToNoiseRatio.from_sfts(
        F0=writer.F0, sftfilepath=writer.sftfilepath
    )
    squared_snr = snr.compute_snr2(
        Alpha=writer.Alpha, 
        Delta=writer.Delta,
        psi=writer.psi,
        phi=writer.phi, 
        h0=writer.h0,
        cosi=writer.cosi
    )
    return np.sqrt(squared_snr)
def GenerateData(num_signals=1,outdir="/home/wangjingqi/input/dataset/g2net",band=0.2,samples=[],append=True):
    os.makedirs(outdir, exist_ok=True)
    # These parameters describe background noise and data format
    writer_kwargs = {
                    "tstart": 1238166018,
                    "duration": 4965*1800,  
                    "detectors": "H1,L1",        
                    "sqrtSX": 1e-23,          
                    "Tsft": 1800,  
                    "Band": band,           
                    "SFTWindowType": "tukey", 
                    "SFTWindowBeta": 0.01,
                }

    # This class allows us to sample signal parameters from a specific population.
    # Implicitly, sky positions are drawn uniformly across the celestial sphere.
    # PyFstat also implements a convenient set of priors to sample a population
    # of isotropically oriented neutron stars.
    signal_parameters_generator = pyfstat.AllSkyInjectionParametersGenerator(
        priors={
            "tref": writer_kwargs["tstart"],
            "F0": lambda: 100,
            "F1": lambda: 10**stats.uniform(-12, 4).rvs(),
            "h0": lambda: writer_kwargs["sqrtSX"] / stats.uniform(1, 50).rvs(),
            **pyfstat.injection_parameters.isotropic_amplitude_priors,
        }
    )

    
    if os.path.exists(os.path.join(outdir,"generted_train_labels.csv")) and append:
        id2label = pd.read_csv(os.path.join(outdir,"generted_train_labels.csv"))
        start = id2label.shape[0]
    else:
        id2label = pd.DataFrame(columns=["id","target"])
        start = 0
    num_signals = 2*num_signals
    for ind in range(start,num_signals,2):

        # Draw signal parameters.
        # Noise can be drawn by setting `params["h0"] = 0
        
        params = signal_parameters_generator.draw()
        writer_kwargs["duration"]=samples[random.randint(0,len(samples)-1)]*1800
        writer_kwargs["outdir"] = os.path.join(outdir,f"Signal_{ind}")
        writer_kwargs["label"] = f"Signal_{ind}"
        writer0 = pyfstat.Writer(**writer_kwargs, **params)
        writer0.make_data()
        squared_snr = get_squared_snr(writer0)
        clear_and_resave(writer0.sftfilepath)
        
        id2label = id2label.append({"id":f"Signal_{ind}","target":squared_snr},ignore_index=True)
        
        params = signal_parameters_generator.draw()
        params["h0"] = 0
        writer_kwargs["duration"]=samples[random.randint(0,len(samples)-1)]*1800
        writer_kwargs["outdir"] = os.path.join(outdir,f"Signal_{ind+1}")
        writer_kwargs["label"] = f"Signal_{ind+1}"
        writer1 = pyfstat.Writer(**writer_kwargs, **params)
        writer1.make_data()
        clear_and_resave(writer1.sftfilepath)
        id2label = id2label.append({"id":f"Signal_{ind+1}","target":0},ignore_index=True)
        id2label.to_csv(os.path.join(outdir,"generted_train_labels.csv"),index=False)
    # Data can be read as a numpy array using PyFstat
    # frequency, timestamps, amplitudes = pyfstat.utils.get_sft_as_arrays(
    #     writer.sftfilepath
    # )
if __name__ == "__main__":
    samples = pd.read_csv("/home/wangjingqi/g2net/timestamps_len.csv")["timestamps_len"].values.tolist()
    GenerateData(num_signals=5000,outdir="/home/wangjingqi/input/dataset/g2net/generted_train",band=0.2,samples=samples,append=False)
