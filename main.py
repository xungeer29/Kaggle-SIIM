import warnings
warnings.filterwarnings(action='ignore')

import time
import os
import numpy as np
import collections
from collections import defaultdict, deque, Counter

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
import albumentations as alb
import visdom

from utils import seed_everything
from lr_schedulers import GradualWarmupScheduler
from train import train_one_epoch
from validation import do_valid
from test import inference, inference_all
from models import DeepLabV3, ResNetUNet
from dataset import SIIMDataset, do_split, SIIMDataset_all, BalanceClassSampler
from loss import DiceLoss, Weight_Soft_Dice_Loss, BCELoss, Weight_BCELoss, FocalLoss, MixedLoss, Lovasz_Loss
from config import config

seed_everything(config.seed)
vis = visdom.Visdom(env=f'Kaggle-SIIM-{config.model}')

if config.model == 'deeplab':
    deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
    if config.upsample == 'CARAFE':
        model_ft = DeepLabV3(deeplabv3, num_classes=1)
    else:
        model_ft = deeplabv3
elif config.model == 'unet':
    backbone = models.resnet34(pretrained=True)
    model_ft = ResNetUNet(backbone, n_class_seg=1, n_class_cls=2)
if os.path.exists(config.init_checkpoint):
    model_ft = torch.load(config.init_checkpoint)
    model_ft.train()
    for param in model_ft.parameters():
        param.requires_grad = True

print(f'\nmodel: {config.model}\nupsample: {config.upsample}')
if config.Gradient_Accumulation:
    print(f'Accumulation Steps: {config.accumulation_steps}')
print(f'Image Size: {config.size}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft.to(device)
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS > 1:
    model_ft = torch.nn.DataParallel(model_ft)
_ = model_ft.to(device)

augs = alb.Compose([
                    alb.HorizontalFlip(p=0.5),
                    alb.CLAHE(p=0.5),
                    alb.CenterCrop(p=0.5, height=int(1024*0.9), width=int(1024*0.9)),
                    alb.OneOf([
                            alb.RandomContrast(),
                            alb.RandomGamma(),
                            alb.RandomBrightness(),
                            ], p=0.3),
                    alb.OneOf([
                            alb.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                            alb.GridDistortion(),
                            alb.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                            ], p=0.3),
#                     alb.PadIfNeeded(p=1, min_height=Size, min_width=Size, border_mode=0)
            ])
# dataset with only mask images
# dataset_train = SIIMDataset(config.train_rle_path, config.train_png_path, augs=augs)
# print('dataset_train:', len(dataset_train))
# data_loader = torch.utils.data.DataLoader(
#     dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True
# )
# dataset with all images
train_ids, train_rles, val_ids, val_rles = do_split(config.train_rle_path)
print('Train dataset:')
# train_ids = train_ids + val_ids
# train_rles = train_rles + val_rles # no validation set
dataset_train = SIIMDataset_all(train_ids, train_rles, config.train_png_path, augs=augs)
data_loader = DataLoader(dataset_train, sampler=BalanceClassSampler(dataset_train, 3*len(dataset_train)),
                         batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
print('Validation dataset:')
dataset_val = SIIMDataset_all(val_ids, val_rles, config.train_png_path, augs=augs)
dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

# construct an optimizer
params = [p for p in model_ft.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()),lr=config.lr)
# optimizer = torch.optim.RMSprop(params, lr=config.lr, alpha = 0.95)
if os.path.exists(config.init_optimizer):
    ckpt = torch.load(config.init_optimizer)
    optimizer.load_state_dict(ckpt['optimizer'])

# lr_scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.3)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs*len(data_loader))
lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=100, 
                                      total_epoch=min(1000, len(data_loader)-1), 
                                      after_scheduler=scheduler_cosine)

# loss function
criterion = DiceLoss()
# criterion = Weight_Soft_Dice_Loss(weight=[0.1, 0.9])
# criterion = BCELoss()
# criterion = MixedLoss(10.0, 2.0)
# criterion = Weight_BCELoss(weight_pos=0.25, weight_neg=0.75)
# criterion = Lovasz_Loss(margin=[1, 5]

print('start training...')
train_start = time.time()
for epoch in range(config.num_epochs):
    epoch_start = time.time()
    model_ft, optimizer = train_one_epoch(model_ft, data_loader, criterion, 
                                          optimizer, lr_scheduler=lr_scheduler, device=device, 
                                          epoch=epoch, vis=vis)
    do_valid(model_ft, dataloader_val, criterion, epoch, device, vis=vis)
    print('Epoch time: {:.3f}min\n'.format((time.time()-epoch_start)/60/60))

print('total train time: {}hours {}min'.format(int((time.time()-train_start)/60//60), int((time.time()-train_start)/60%60)))
inference_all(model_ft, device=device)
inference(model_ft, device=device)
torch.save(model_ft, f'{config.model}.pth')
torch.save({'optimizer': optimizer.state_dict(),
            'epoch': epoch,}, 
            'optimizer.pth')