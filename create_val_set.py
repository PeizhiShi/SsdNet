from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import numpy as np
import cupy as cp
import csv
import matplotlib.pyplot as plt
import random

num_samples = 1100

random.seed(1024)
partition = create_partition(512, 100)
list_IDs = partition['val']



DIR = 'data/ValSet/'
start_counter = int(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])/2)
random.seed(start_counter)     

print(start_counter)      


with open('data/minfo.csv', mode='r') as infile:
    reader = csv.reader(infile)
    list_size = {rows[0]:int(rows[1]) for rows in reader}

counter = start_counter
for idx in range(num_samples):
    rotation = int(idx%6)
    m_idx = int(idx/6)
    if rotation == 0:
        cur_model, cur_model_label, cur_model_components = achieve_random_model(list_IDs, list_size)
            
    img, _ = create_img(cur_model, rotation, True)        
    target = achieve_model_gt(cur_model_label, cur_model_components, rotation)
    
    if target.shape[0] == 0:
        continue
    
    print('processing the', idx, 'th image...')
    filename = DIR+str(counter)
    plt.imsave(filename+'.png',img,cmap='gray',vmin=0,vmax=255)

    
    np.save(filename+'.npy',target)
    counter = counter + 1
