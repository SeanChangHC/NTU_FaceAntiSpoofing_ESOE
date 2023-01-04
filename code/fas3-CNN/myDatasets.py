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
def get_OULU_train_val_set(root, ratio=0.9, cv=0):
    
        
 #import image files path into a list 
    images = glob(root+'/*')


#extract label from filename to a list
    labels = []
    for root, dirs, files in os.walk(root):
        for filename in files:
            # if 'live' in filename:
            #     labels.append(int(1))
            if 'live' in filename:
                labels.append(int(1))                
            # elif '_1.' in filename:
            #     labels.append(int(1))
            else:
                labels.append(0)


    info = np.stack( (np.array(images), np.array(labels)) ,axis=1)

    # print(info)
    N = info.shape[0]

    # apply shuffle to generate random results 
    np.random.shuffle(info)
    x = int(N*ratio) 

    all_images, all_labels = info[:,0].tolist(), info[:,1].astype(np.int32).tolist()


    train_image = all_images[:x]
    val_image = all_images[x:]

    train_label = all_labels[:x] 
    val_label = all_labels[x:]

    

    
    ## TO DO ## 
    # Define your own transform here 
    # It can strongly help you to perform data augmentation and gain performance
    # ref: https://pytorch.org/vision/stable/transforms.html
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    transform_set = [ 
             transforms.CenterCrop(224), 
            #  transforms.Pad(100, padding_mode='symmetric'),
             transforms.RandomRotation(30),
             transforms.ColorJitter()
            ]
    
    train_transform = transforms.Compose([
                
                transforms.Resize((256, 256)),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2),
                transforms.RandomCrop((224, 224)),	
                transforms.ColorJitter(brightness=(0, 5), contrast=(0, 5), saturation=(0, 5), hue=(-0.1, 0.1)),            
                transforms.RandomAffine(degrees=20, translate=(0, 0.2), scale=(0.9, 1.1), shear=(6, 9)),
                transforms.RandomApply(transform_set, p=0.5), 
                transforms.ToTensor(),
                transforms.Normalize(mean = means,std = stds)
            ])
  
    
    val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

  
    ## TO DO ##
    # Complete class cifiar10_dataset
    train_set, val_set = cifar10_dataset(images=train_image, labels=train_label,transform=train_transform), \
                        cifar10_dataset(images=val_image, labels=val_label,transform=val_transform)



    return train_set, val_set



## TO DO ##
# Define your own cifar_10 dataset
class cifar10_dataset(Dataset):
    def __init__(self, images, labels=None, transform=None, prefix = './p2_data/train'):
        
        # It loads all the images' file name and correspoding labels here
        self.images = images 
        self.labels = labels 
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        # self.prefix = prefix
        
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ## TO DO ##
        img_name = self.images[idx]
        # img_name = self.prefix + '/' + img_name
        image = Image.open(img_name).convert('RGB')
        if self.transform:
          image = self.transform(image)
        return image, self.labels[idx]
        
