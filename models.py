import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models

from CARAFE import CARAFE
from config import config

# UNet-ResNet34
def l2_norm(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):

    def __init__(self, backbone, n_class_seg=1, n_class_cls=2):
        super().__init__()
        
        self.base_model = backbone
        
        self.base_layers = list(backbone.children())                
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        # self.upsample0 = CARAFE(64, 64)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        # self.upsample1 = CARAFE(64, 64)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        # self.upsample2 = CARAFE(128, 128)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        
        if config.upsample == 'carafe':
	        self.upsample0 = CARAFE(128, 128)
	        self.upsample1 = CARAFE(256, 256)
	        self.upsample2 = CARAFE(256, 256)
	        self.upsample3 = CARAFE(512, 512)
	        self.upsample4 = CARAFE(512, 512)
        elif config.upsample == 'bilinear':
        	self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, n_class_seg, 1)
        
#         # for classifacation
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.logit_cls = nn.Conv2d(512, n_class_cls, kernel_size=1, stride=1, padding=0)
        
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        c = layer4
        
        layer4 = self.layer4_1x1(layer4)
        if config.upsample == 'bilinear':
        	x = self.upsample(layer4)
        elif config.upsample == 'carafe':
        	x = self.upsample4(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        if config.upsample == 'bilinear':
        	x = self.upsample(x)
        elif config.upsample == 'carafe':
        	x = self.upsample3(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        if config.upsample == 'bilinear':
        	x = self.upsample(x)
        elif config.upsample == 'carafe':
        	x = self.upsample2(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        if config.upsample == 'bilinear':
        	x = self.upsample(x)
        elif config.upsample == 'carafe':
        	x = self.upsample1(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        if config.upsample == 'bilinear':
        	x = self.upsample(x)
        elif config.upsample == 'carafe':
        	x = self.upsample0(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        logit_seg = self.conv_last(x)
        
#         # for classification
#         c = self.avgpool(c)
#         c = self.logit_cls(c)
#         logit_cls = c.view(c.size(0), -1)
#         logit_cls = l2_norm(logit_cls)
        
        return logit_seg#, logit_cls

class DeepLabV3(nn.Module):
    def __init__(self, backbone, num_classes=1):
        super().__init__()
        self.backbone = backbone
        if config.upsample == 'carafe':
        	self.upsample = CARAFE(inC=256, outC=num_classes, up_factor=8)
        elif config.upsample == 'bilinear':
        	self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')
        self.model = nn.Sequential(*list(self.backbone.children()))
        
    def forward(self, x): # [1, 3, 1024, 1024]
        x = self.backbone.backbone.conv1(x)
        x = self.backbone.backbone.bn1(x)
        x = self.backbone.backbone.relu(x)
        x = self.backbone.backbone.maxpool(x)#; print(f'layer0: {x.size()}') # [1, 64, 256, 256]  downsample x4

        x = self.backbone.backbone.layer1(x)#; print(f'layer1: {x.size()}') # [1, 256, 256, 256]
        x = self.backbone.backbone.layer2(x)#; print(f'layer2: {x.size()}') # [1, 512, 128, 128]  downsample x2
        x = self.backbone.backbone.layer3(x)#; print(f'layer3: {x.size()}') # [1, 1024, 128, 128]
        x = self.backbone.backbone.layer4(x)#; print(f'layer4: {x.size()}') # [1, 2048, 128, 128]
        
        x = self.backbone.classifier[0](x)#; print(f'ASPP: {x.size()}') # [1, 256, 128, 128]
        x = self.backbone.classifier[1](x)#; print(f'ASPP.conv:ReLU: {x.size()}') # [1, 256, 128, 128]
        x = self.backbone.classifier[2](x)#; print(f'ASPP.BN: {x.size()}') # [1, 256, 128, 128]
        x = self.backbone.classifier[3](x)#; print(f'ASPP.ReLU: {x.size()}') # [1, 256, 128, 128]
        
        if config.upsample == 'carafe':
        	logits = self.upsample(x)
        
        if config.upsample == 'bilinear':
        	x = self.backbone.classifier[4](x)
        	logits = self.upsample(x)
        	logits = logits['out'][:, 1, :, :]
        	logits = torch.unsqueeze(logits, dim=1)
        
        # logits = self.backbone.classifier(x)
        return logits


if __name__ == '__main__':
	data = torch.rand((4, 3, 256, 256))

	# test ResNetUNet
	backbone = models.resnet34(pretrained=True)
	model = ResNetUNet(backbone, n_class_seg=1, n_class_cls=2).cuda()
	print(model)
	out1 = model(data.cuda())
	print(f'ResNetUNet: {out1.size()}')

	# test deeplabV3-resnet50
	backbone = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
	model = DeepLabV3(backbone)
	print(model)
	model = model.cuda()
	logits = model(data.cuda())
	print(f'DeepLabV3: {logits.size()}')

	backbone.cuda()
	logits = backbone(data.cuda())
	logits = logits['out'][:, 1, :, :]
	logits = torch.unsqueeze(logits, dim=1)
	print(f'deeplabv3_resnet50: {logits.size()}')