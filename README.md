# Kaggle-SIIM

## Discussion
* [Dmytro Panchenko:How is metric calculated?](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97225#latest-563443):讲解指标的计算方式
* [Heng CherKeng: starter kit](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97456#latest-563494): 收集各种分割的trcik并进行复现
* [Previous segmentation challenges](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/96992#latest-563339):以前的分割竞赛 [Resources | Papers With (really good) Code](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97198#latest-561177)
* [Optimizing Dice is Harder than IoU](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97474#latest-563400): 讨论IoU是不是子模函数 [Yes, IoU loss is submodular – as a function of the mispredictions](https://arxiv.org/pdf/1809.01845.pdf) [INTERSECTION OVER UNION IS NOT A SUBMODULAR FUNCTION](https://arxiv.org/pdf/1809.00593.pdf)
* [Heng CherKeng: baseline methods](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97518#latest-562789):包含肺部检测的论文和代码
* [Timestamp in file names ?](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97119#latest-562337):发现文件名是时间戳
  ```python
  from datetime import datetime
  ts = 1517875163.537053
  print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

  2018-02-05 23:59:23
  ```
 
## Kernel
* [EDA](https://www.kaggle.com/unvirtual/eda-of-training-test-data/notebook):
* [Image+Mask Augmentations--pytorch](https://www.kaggle.com/abhishek/image-mask-augmentations):图像分割的数据扩充方法
* [dataset:DICOM + MASK+ PNG + FHIR](https://www.kaggle.com/anisayari/siimacrpneumothoraxsegmentationzip-dataset)：处理好的数据集，包含mask，png等
* [First steps with SIIM-ACR Pneumothorax Data](https://www.kaggle.com/steubk/first-steps-with-siim-acr-pneumothorax-data/comments?scriptVersionId=16473604#The-bimodal-mean_pixel_value-distribution):数据分布的探索
* [mask-rcnn with augmentation and multiple masks](https://www.kaggle.com/abhishek/mask-rcnn-with-augmentation-and-multiple-masks/notebook): pytorch mask-rcnn的训练， LB=0.8042
* [Visualizing Submission File](https://www.kaggle.com/abhishek/visualizing-submission-file): 可视化mask

## Discover
* 肺部吸入与呼出空气使图像灰度直方图有两个峰值

  ![像素均值计算](./figs/histogram.png)
  
* 数据集有10712张图像，但是只有10675张图像有标注，有mask的图像有3286张，大部分只有1个mask
  ![mask数量统计](./figs/nMasks.png)
  
* mask占比
  ![](./figs/mask_coverage.png)
  
## Experiment
* Mask-RCNN

| 方法 | 结果 | 备注 |
| :------| ------: | :------: |
| baseline | 0.8042 | epoch = 5 |
| RandomColor | 0.7888 | down |
| epoch: 5->20 | 0.8054 |  |
| epoch:20, vertical_flip | 0.8043 | down |
| epoch:20, rotation | 0.7988 | down |
| epoch:20, clahe | 0.8043 | down |
