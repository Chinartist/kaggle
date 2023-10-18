import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2

#设置特定GPU的环境变量
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.types import DALIDataType
from pydicom.filebase import DicomBytesIO
from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type


def convert_dicom_to_j2k(file, save_folder=""):
    patient = file.split('/')[-2]
    image = file.split('/')[-1][:-4]
    dcmfile = pydicom.dcmread(file)

    if dcmfile.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':
        with open(file, 'rb') as fp:
            raw = DicomBytesIO(fp.read())
            ds = pydicom.dcmread(raw)
        offset = ds.PixelData.find(b"\x00\x00\x00\x0C")  #<---- the jpeg2000 header info we're looking for
        hackedbitstream = bytearray()
        hackedbitstream.extend(ds.PixelData[offset:])
        with open(save_folder + f"{patient}_{image}.jp2", "wb") as binary_file:
            binary_file.write(hackedbitstream)

            
@pipeline_def
def j2k_decode_pipeline(j2kfiles):
    jpegs, _ = fn.readers.file(files=j2kfiles)
    images = fn.experimental.decoders.image(jpegs, device='mixed', output_type=types.ANY_DATA, dtype=DALIDataType.UINT16)
    return images


def cut_off(img):
    X = img
    # Some images have narrow exterior "frames" that complicate selection of the main data. Cutting off the frame
    X = X[5:-5, 5:-5]

    # regions of non-empty pixels
    output= cv2.connectedComponentsWithStats((X > 0.05).astype(np.uint8)[:, :], 8, cv2.CV_32S)

    # stats.shape == (N, 5), where N is the number of regions, 5 dimensions correspond to:
    # left, top, width, height, area_size
    stats = output[2]
    # finding max area which always corresponds to the breast data. 
    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]
    x2 = x1 + w
    y2 = y1 + h
    # cutting out the breast data
    X_fit = X[y1: y2, x1: x2]

    return X_fit
import dicomsdl

def dicomsdl_to_numpy_image(dicom, index=0):
    info = dicom.getPixelDataInfo()
    dtype = info['dtype']
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]
    outarr = np.empty(shape, dtype=dtype)
    dicom.copyFrameData(index, outarr)
    return outarr

def load_img_dicomsdl(f):
    return dicomsdl_to_numpy_image(dicomsdl.open(f))

def process(f, img_size=None, save_folder=""):
    patient = f.split('/')[-2]
    image = f.split('/')[-1][:-4]
    dicom = pydicom.dcmread(f)
    if dicom.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':  # ALREADY PROCESSED
        return

    try:
        img = load_img_dicomsdl(f)
    except:
        img = dicom.pixel_array

    img = (img - img.min()) / (img.max() - img.min())

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    img = cut_off(img)
    if img_size is not None:
        img = cv2.resize(img, (img_size[1],img_size[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(save_folder + f"{patient}_{image}.png", (img * 255).astype(np.uint8))


from joblib import Parallel, delayed
from tqdm import tqdm
import os
import shutil
import glob
SAVE_FOLDER = '/home/wangjingqi/input/dataset/rsna/images/'
J2K_FOLDER = "/home/wangjingqi/input/dataset/rsna/j2k/"
if os.path.exists(SAVE_FOLDER):
    shutil.rmtree(SAVE_FOLDER)
if os.path.exists(J2K_FOLDER):
    shutil.rmtree(J2K_FOLDER)
os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(J2K_FOLDER, exist_ok=True)


IMG_PATH = "/home/wangjingqi/input/dataset/rsna/train_images/"
train_images = glob.glob(f"{IMG_PATH}*/*.dcm")

if len(train_images) > 100:
    N_CHUNKS = 30
else:
    N_CHUNKS = 1

CHUNKS = [(len(train_images) / N_CHUNKS * k, len(train_images) / N_CHUNKS * (k + 1)) for k in range(N_CHUNKS)]
CHUNKS = np.array(CHUNKS).astype(int)
print(CHUNKS)


for chunk in tqdm(CHUNKS):
    os.makedirs(J2K_FOLDER, exist_ok=True)

    _ = Parallel(n_jobs=2)(
        delayed(convert_dicom_to_j2k)(img, save_folder=J2K_FOLDER)
        for img in train_images[chunk[0]: chunk[1]]
    )
    
    j2kfiles = glob.glob(J2K_FOLDER + "*.jp2")

    if not len(j2kfiles):
        continue

    pipe = j2k_decode_pipeline(j2kfiles, batch_size=1, num_threads=2, device_id=0, debug=True)
    pipe.build()

    for i, f in enumerate(j2kfiles):
        patient, image = f.split('/')[-1][:-4].split('_')
        dicom = pydicom.dcmread(IMG_PATH + f"{patient}/{image}.dcm")

        out = pipe.run()

        # Dali -> Torch
        img = out[0][0]
        img_torch = torch.empty(img.shape(), dtype=torch.int16, device="cuda")
        feed_ndarray(img, img_torch, cuda_stream=torch.cuda.current_stream(device=0))
        img = img_torch.float()

        # Scale, resize, invert on GPU !
        min_, max_ = img.min(), img.max()
        img = (img - min_) / (max_ - min_)

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img

        # Back to CPU + SAVE
        img = img.cpu().numpy()
        img = cut_off(img)
        cv2.imwrite(SAVE_FOLDER + f"{patient}_{image}.png", (img * 255).astype(np.uint8))
    shutil.rmtree(J2K_FOLDER)



_ = Parallel(n_jobs=2)(
    delayed(process)(img, img_size=None, save_folder=SAVE_FOLDER)
    for img in tqdm(train_images)
)