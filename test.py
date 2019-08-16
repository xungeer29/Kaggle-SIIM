import os
import time
import PIL
from PIL import Image, ImageFile
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torchvision import transforms

import albumentations as alb
from skimage import measure, color, morphology

from config import config
from utils import mask_to_rle

# inference all images, without leak informations
def inference_all(model_ft, device):
    model_ft.eval()
    for param in model_ft.parameters():
        param.requires_grad = False
    model_ft.to(device)
    assert model_ft.training == False
    
    test_start = time.time()
    sample_df = pd.read_csv(config.sub_sample_path)

    tt = transforms.ToTensor()
    sublist = []
    counter = 0
    best_thr = 0.5
    Noise_th = 75.0*(1024/128.0)**2
    for index, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc='test'):
        image_id = row['ImageId']
        img_path = os.path.join(config.test_png_path, image_id + '.png')

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        img_row = img.resize((1024, 1024), resample=Image.BILINEAR)
        img = tt(img_row)
        img = img.reshape((1, *img.numpy().shape))
        logits = model_ft(img.to(device))[:, 0, :, :]
        preds = torch.sigmoid(logits)[0].cpu().numpy()
#         mask = (preds > best_thr).astype(np.uint8) # if no_tta

        # TTA HFlip
        hflip_img = img.flip(-1)
        hflip_logits = model_ft(hflip_img.to(device))[:, 0, :, :]
        hflip_logits = hflip_logits.flip(-1)
        hflip_preds = torch.sigmoid(hflip_logits)[0].cpu().numpy()

        # CLAHE TTA
        clahe = alb.CLAHE(p=1.0)
        clahe_im = clahe(image=np.asarray(img_row))['image']
        clahe_im = tt(clahe_im)
        clahe_im = clahe_im.reshape((1, *clahe_im.numpy().shape))
        clahe_logits = model_ft(clahe_im.to(device))[:, 0, :, :]
        clahe_preds = torch.sigmoid(clahe_logits)[0].cpu().numpy()

        preds = (preds+hflip_preds+clahe_preds) / 3
        mask = (preds > best_thr).astype(np.uint8)

        mask = Image.fromarray(np.uint8(mask))
        mask = mask.resize((1024, 1024), resample=Image.BILINEAR)
        mask = np.asarray(mask)
        mask = morphology.remove_small_objects(mask, min_size=Noise_th, connectivity=2)

#         mask = mask*255
#         kernel = np.ones((6,6), np.uint8)
#         erode_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=2)
#         mask = mask / 255

        mask = mask.T
        if np.count_nonzero(mask) == 0:
            rle = " -1"
        else:
            rle = mask_to_rle(mask, width, height)
        if image_id == '1.2.276.0.7230010.3.1.4.8323329.6830.1517875201.424165':
            rle = '-1'
        sublist.append([image_id, rle])

    submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
    if not os.path.exists('results'):
        os.makedirs('results')
    submission_df.to_csv("./results/sub_all_images.csv", index=False)
    print('inference all images test time: {:.3f}min'.format((time.time()-test_start)/60))

def inference(model_ft, device):
    model_ft.eval()
    for param in model_ft.parameters():
        param.requires_grad = False
    model_ft.to(device)
    assert model_ft.training == False
    test_start = time.time()
    sample_df = pd.read_csv(config.sub_sample_path)
    leak_df = pd.read_csv(config.leak_path)
    leak_id = []
    for id, anno in zip(leak_df['ImageId'], leak_df['EncodedPixels']):
        if anno =='-1':
            leak_id.append(id)
    print(f'leak_id: {len(leak_id)}')
    print(f'leak id: {leak_id[:5]}')
    tt = transforms.ToTensor()
    sublist = []
    counter = 0
    best_thr = 0.5
    Noise_th = 75.0*(1024/128.0)**2
    for index, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc='test'):
        image_id = row['ImageId']
        if image_id not in leak_id:
            img_path = os.path.join(config.test_png_path, image_id + '.png')

            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            img_row = img.resize((1024, 1024), resample=Image.BILINEAR)
            img = tt(img_row)
            img = img.reshape((1, *img.numpy().shape))
            logits = model_ft(img.to(device))[:, 0, :, :]
            preds = torch.sigmoid(logits)[0].cpu().numpy()
    #         mask = (preds > best_thr).astype(np.uint8) # if no_tta

            # TTA HFlip
            hflip_img = img.flip(-1)
            hflip_logits = model_ft(hflip_img.to(device))[:, 0, :, :]
            hflip_logits = hflip_logits.flip(-1)
            hflip_preds = torch.sigmoid(hflip_logits)[0].cpu().numpy()

            # CLAHE TTA
            clahe = alb.CLAHE(p=1.0)
            clahe_im = clahe(image=np.asarray(img_row))['image']
            clahe_im = tt(clahe_im)
            clahe_im = clahe_im.reshape((1, *clahe_im.numpy().shape))
            clahe_logits = model_ft(clahe_im.to(device))[:, 0, :, :]
            clahe_preds = torch.sigmoid(clahe_logits)[0].cpu().numpy()

            preds = (preds+hflip_preds+clahe_preds) / 3
            mask = (preds > best_thr).astype(np.uint8)

            mask = Image.fromarray(np.uint8(mask))
            mask = mask.resize((1024, 1024), resample=Image.BILINEAR)
            mask = np.asarray(mask)
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

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)
    submission_df.to_csv("results/sub_ony_mask.csv", index=False)
    print('test time: {:.3f}min'.format((time.time()-test_start)/60))
    torch.save({'model': model_ft, 'leaks': leak_id}, 'checkpoints/deeplabv3_leak.pth')