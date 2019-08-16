import os
import random
import cv2
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections
from collections import deque, Counter

import torch
from torch.utils.data.sampler import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import albumentations as alb

from config import config
from utils import rle2mask

import warnings
warnings.filterwarnings(action='ignore')


# only mask images
class SIIMDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, img_dir, augs=None):
        self.df = pd.read_csv(df_path)
        df_dict = {}
        for id, pix in zip(self.df['ImageId'], self.df[' EncodedPixels']):
            if id not in df_dict.keys():
                df_dict[id] = pix
            else:
                df_dict[id] =  df_dict[id]+' '+pix
        self.df = pd.DataFrame(list(df_dict.items()), columns=self.df.columns.values)
        self.df = self.df[self.df[' EncodedPixels'] != ' -1']

        self.height = config.size
        self.width = config.size
        self.image_dir = img_dir
        self.image_info = collections.defaultdict(dict)
        self.augs = augs

        counter = 0
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_id = row['ImageId']
            image_path = os.path.join(self.image_dir, image_id)
            if os.path.exists(image_path + '.png') and row[" EncodedPixels"].strip() != "-1" and image_id != '1.2.276.0.7230010.3.1.4.8323329.12495.1517875239.386976':
                self.image_info[counter]["image_id"] = image_id
                self.image_info[counter]["image_path"] = image_path
                self.image_info[counter]["annotations"] = row[" EncodedPixels"].strip()
                counter += 1

    def __getitem__(self, idx):
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path + '.png').convert("RGB")
        width, height = img.size
        info = self.image_info[idx]

        mask = rle2mask(info['annotations'], width, height)
        mask = mask.T
        
#         mask = np.expand_dims(mask, axis=0)

#         mask = mask.T*255
#         kernel = np.ones((6,6), np.uint8)
#         dilation_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2) / 255
# #         mask = Image.fromarray(dilation_mask/255)

        if self.augs is not None:
            img = np.array(img)
            mask = np.array(mask)
#             size = random.randint(512,1024)
#             img = cv2.resize(img, (config.size, config.size))
#             mask = cv2.resize(mask, (config.size, config.size))
            augmented = self.augs(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = cv2.resize(img, (config.size, config.size))
        mask = cv2.resize(mask, (config.size, config.size))
        
        mask = torch.as_tensor(mask, dtype=torch.float)

        img = transforms.ToTensor()(img)
#         img = transforms.Normalize(mean=[0.491, 0.491, 0.491], 
#                                    std=[0.249, 0.249, 0.249])(img)
        
        return img, mask

    def __len__(self):
        return len(self.image_info)
    

# for all images
# split
def do_split(csv):
    df = pd.read_csv(csv)
    ids, rles = [], []
    for id, rle in zip(df['ImageId'], df[' EncodedPixels']):
        id = id.strip()
        rle = rle.strip()
        if id not in ids:
            ids.append(id)
            rles.append(rle)
        else:
            idx = ids.index(id)
            rles[idx] = rles[idx] + ' '+ rle
    val_ids = random.sample(ids, 600)
    val_rles = [rles[ids.index(id)] for id in val_ids]
    train_ids = [id for id in ids if id not in val_ids]
    train_rles = [rles[ids.index(id)] for id in train_ids]
    
    return train_ids, train_rles, val_ids, val_rles

class SIIMDataset_all(torch.utils.data.Dataset):
    def __init__(self, ids, rles, img_dir, augs=None):
        self.image_dir = img_dir
        self.augs = augs

        self.ids, self.rles, self.labels = [], [], []
        for id, rle in zip(ids, rles):
            if id == '1.2.276.0.7230010.3.1.4.8323329.12495.1517875239.386976':
                continue
            image_path = os.path.join(self.image_dir, id)
            if os.path.exists(image_path + '.png'):
                self.ids.append(id)
                self.rles.append(rle)
                if rle.strip() == '-1':
                    self.labels.append(0)
                else:
                    self.labels.append(1)
        assert len(self.ids)==len(self.labels)==len(self.rles)
        counter = Counter(self.labels)
        print(f'  Total images:{len(ids)}\n  Pneumothorax images:{counter[1]}\n  Health images:{counter[0]}\n')      

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.ids[idx])
        img = Image.open(img_path + '.png').convert("RGB")
        width, height = img.size

        mask = rle2mask(self.rles[idx], width, height)
        mask = mask.T
        
        label = self.labels[idx]

        if self.augs is not None:
            augmented = self.augs(image=np.array(img), mask=np.array(mask))
            img = augmented['image']
            mask = augmented['mask']

        img = cv2.resize(img, (config.size, config.size))
        mask = cv2.resize(mask, (config.size, config.size))
#         mask = cv2.resize(mask, (int(Size/4), int(Size/4))) # ********* the output of network is image_size/4 ***************************
        
        mask = torch.as_tensor(mask, dtype=torch.float)
        img = transforms.ToTensor()(img)
#         img = transforms.Normalize(mean=[0.491, 0.491, 0.491], 
#                                    std=[0.249, 0.249, 0.249])(img)
        
        return img, mask, label, self.ids[idx]

    def __len__(self):
        return len(self.ids)
    
class BalanceClassSampler(Sampler):

    def __init__(self, dataset, length=None):
        self.dataset = dataset

        if length is None:
            length = len(self.dataset)

        self.length = length

    def __iter__(self):
        pos_index = np.array([i for i, label in enumerate(self.dataset.labels) if label==1])
        neg_index = np.array([i for i, label in enumerate(self.dataset.labels) if label==0])
        
        half = self.length//2 + 1
        pos = np.random.choice(pos_index, half, replace=True)
        neg = np.random.choice(neg_index, half, replace=True)
        l = np.stack([pos,neg]).T
        l = l.reshape(-1)
        l = l[:self.length]
        return iter(l)

    def __len__(self):
        return self.length



if __name__ == '__main__':
    augs = alb.Compose([
#                        alb.CenterCrop(p=1, height=int(1024*0.9), width=int(1024*0.9)),
                       alb.HorizontalFlip(p=1),
                       alb.CLAHE(p=0.5),
                       alb.PadIfNeeded(p=1, min_height=config.size, min_width=config.size, border_mode=0)
                       ])
    print('Test SIIMDataset...')
    dataset_train = SIIMDataset(config.train_rle_path, config.train_png_path, augs=augs)
    dataloader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0)
    for i, (im, mask) in enumerate(dataloader):
        if mask.view(mask.size(0), -1).sum(-1) != 0:            
            img = im[0][0].view(config.size, config.size).numpy()
            img = (img*255).astype(np.uint8)
            img = PIL.Image.fromarray(img).convert('RGB')
            
            mask = mask.view(config.size, config.size).numpy().astype(np.uint8)
            mask = mask*255
            
#             print(mask.shape, mask)
#             mask = PIL.Image.fromarray(mask).resize((1024,1024))
            fig = plt.figure(figsize=(15, 10))
            a = fig.add_subplot(1, 3, 1)
            plt.imshow(img, cmap='bone') #original x-ray
            a.set_title("Original x-ray image")
            plt.grid(False)
            plt.axis("off")
            
            a = fig.add_subplot(1, 3, 2)
            imgplot = plt.imshow(mask, cmap='binary')
            a.set_title("The mask")
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            a = fig.add_subplot(1, 3, 3)
            plt.imshow(img, cmap='bone')
            plt.imshow(mask, cmap='binary', alpha=0.3)
            a.set_title("Mask on the x-ray")
            plt.show()
            break

    print('Test SIIMDataset_all...')
    train_ids, train_rles, val_ids, val_rles = do_split(config.train_rle_path)
    dataset_train = SIIMDataset_all(train_ids, train_rles, config.train_png_path, augs=augs)
    dataloader = DataLoader(dataset_train, sampler=BalanceClassSampler(dataset_train), batch_size=1, shuffle=False, num_workers=0)
    for i, (im, mask, label, id) in enumerate(dataloader):
#         if mask.view(mask.size(0), -1).sum(-1) != 0:            
            img = im[0][0].view(config.size, config.size).numpy()
            img = (img*255).astype(np.uint8)
            img = PIL.Image.fromarray(img).convert('RGB')
            
            mask = mask.view(config.size, config.size).numpy().astype(np.uint8)
            mask = mask*255
            
            print(f'label: {label}\tid:{id}')
            
#             print(mask.shape, mask)
#             mask = PIL.Image.fromarray(mask).resize((1024,1024))
            fig = plt.figure(figsize=(15, 10))
            a = fig.add_subplot(1, 3, 1)
            plt.imshow(img, cmap='bone') #original x-ray
            a.set_title("Original x-ray image")
            plt.grid(False)
            plt.axis("off")
            
            a = fig.add_subplot(1, 3, 2)
            imgplot = plt.imshow(mask, cmap='binary')
            a.set_title("The mask")
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            a = fig.add_subplot(1, 3, 3)
            plt.imshow(img, cmap='bone')
            plt.imshow(mask, cmap='binary', alpha=0.3)
            a.set_title("Mask on the x-ray")
            plt.show()
            break