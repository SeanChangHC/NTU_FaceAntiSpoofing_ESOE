import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np 
from torchvision.transforms import transforms
from PIL import Image
import json 
from random import shuffle
from glob import glob
from re import search

root = '/home/fas3/example-img/SiW/protocol3/test'

#import video files path into a list 
videos = glob(root+'/*')


#extract label from filename to a list
labels = []
for root, dirs, files in os.walk(root):
    for filename in files:
        if 'live' in filename:
            labels.append(int(1))
        # elif '_1.' in filename:
        #     labels.append(int(1))
        else:
            labels.append(0)
        
# print(labels)      

        
        
# #extract label from filename to a list
# labels = []
# for root, dirs, files in os.walk(root):
#     for filename in files:
#         labels.append(int(filename[-5]))
    
info = np.stack( (np.array(videos), np.array(labels)) ,axis=1)

print(info[20:])

# N = info.shape[0]


# apply shuffle to generate random results 
# np.random.shuffle(info)
# x = int(N*ratio) 

# all_videos, all_labels = info[:,0].tolist(), info[:,1].astype(np.int32).tolist()


# train_video = all_videos[:x]
# val_video = all_videos[x:]

# train_label = all_labels[:x] 
# val_label = all_labels[x:]


