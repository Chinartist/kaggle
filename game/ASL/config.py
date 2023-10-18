import numpy as np
max_length = 384

num_head   = 8
num_block  = 1
ls = 0.5
seed = 42
num_class  = 250
embed_dim  = 512
hidden_dim = 128
layerdrop  = 0.0
n=16
fd = 0.0
fd1 = 0.4
device = [1]
p=0.5
dropout = 0.3
ROWS_PER_FRAME = 543
nw=4
used_folds = [2]
optim_type = "adamw"
act_name = "quickgelu"
col = ["x","y"]
LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]

LIP_new1 = [13,312,311,310,415,308,324,318,402,317,313,18,14,87,83,178,182,88,85,78,191,80,81,82]

LIP_new2 = [12,268,271,272,407,292,325,319,406,403,316,15,86,179,89,106,96,62,183,42,41,38]

LIP_new3 = [11,302,303,304,408,306,307,320,404,315,16,65,180,90,77,76,184,74,73,72]

LIP_new4 = [0,267,269,270,409,291,375,335,321,405,314,421,17,200,84,201,181,194,91,146,43,61,185,40,39,37]

LIP = list(set(LIP_new1+LIP_new2+LIP_new3+LIP_new4+LIP))
SPOSE = [500, 502, 504, 501, 503, 505, 512, 513]
POSE=np.arange(489, 522).tolist()
REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]
NOSE=[
    1,2,98,327
]
LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()

assert len(LHAND) == len(RHAND)
pointset1 = LIP + LHAND + RHAND#+SPOSE
pointset2 = list(set(np.arange(0, ROWS_PER_FRAME).tolist())-set(pointset1))

num_point1 = len(pointset1)
num_point2 = len(pointset2)

print("num_point1: ", num_point1, "num_point2: ", num_point2,"layerdrop_rate: ", layerdrop,"used_folds",used_folds)

num_hand_feats = len(LHAND)*len(LHAND)
num_main_feats = num_point1*(2*n+2)
input_dim =num_point1*2,len(LHAND)*2*n*2,len(LHAND)*len(LHAND)*2*2