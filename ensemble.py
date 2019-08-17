import os
import cv2
import PIL
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms

import albumentations as alb
from skimage import measure, color, morphology

from config import config
from utils import mask_to_rle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = []
for pth in os.listdir('checkpoints'):
	if 'model' in pth:
		model = torch.load(os.path.join('checkpoints', pth))
		model.eval()
		for param in model.parameters():
		    param.requires_grad = False
		model.to(device)
		models.append(model)

sample_df = pd.read_csv(config.sub_sample_path)
leak_df = pd.read_csv(config.leak_path)
leak_id = []
for id, anno in zip(leak_df['ImageId'], leak_df['EncodedPixels']):
    if anno =='-1':
        leak_id.append(id)
tt = transforms.ToTensor()
sublist = []
counter = 0
best_thr = 0.5
Noise_th = 4650
for index, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc='test'):
    image_id = row['ImageId']
    if image_id not in leak_id:
        img_path = os.path.join(config.test_png_path, image_id + '.png')

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        img_row = img.resize((1024, 1024), resample=Image.BILINEAR)
        img = tt(img_row)
        img = img.reshape((1, *img.numpy().shape))

        for i, model in enumerate(models):
        	logit = model(img.to(device))[:, 0, :, :]
        	pred = torch.sigmoid(logit)[0].cpu().numpy()
        	if i == 0:
        		preds = pred
        	else:
        		preds += pred
        preds = preds / len(models)
        
        # HFlip TTA
        hflip_img = img.flip(-1)
        for i, model in enumerate(models):
        	hflip_logit = model(hflip_img.to(device))[:, 0, :, :]
        	hflip_pred = torch.sigmoid(hflip_logit)[0].cpu().numpy()
        	if i == 0:
        		hflip_preds = hflip_pred
        	else:
        		hflip_preds += hflip_pred
        hflip_preds = hflip_preds / len(models)
        
        # CLAHE TTA
        clahe = alb.CLAHE(p=1.0)
        clahe_im = clahe(image=np.asarray(img_row))['image']
        clahe_im = tt(clahe_im)
        clahe_im = clahe_im.reshape((1, *clahe_im.numpy().shape))
        for i, model in enumerate(models):
        	clahe_logit = model(clahe_im.to(device))[:, 0, :, :]
        	clahe_pred = torch.sigmoid(clahe_logit)[0].cpu().numpy()
        	if i == 0:
        		clahe_preds = clahe_pred
        	else:
        		clahe_preds += clahe_pred
        clahe_preds = clahe_preds / len(models)
        
    #     mask = (preds > 0.5).astype(np.uint8).T
        preds = (preds+hflip_preds+clahe_preds) / 3
        mask = (preds > best_thr).astype(np.uint8)
        mask = morphology.remove_small_objects(mask, min_size=Noise_th, connectivity=2)
        mask = mask.T
        if np.count_nonzero(mask) == 0:
            rle = " -1"
        else:
            rle = mask_to_rle(mask, width, height)
    else:
        rle = '-1'
    if image_id == '1.2.276.0.7230010.3.1.4.8323329.6830.1517875201.424165':
        rle = '-1'
    sublist.append([image_id, rle])

submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
submission_df.to_csv("sub_ensemble.csv", index=False)