# -*- coding:utf-8 -*- 
class DefaultConfigs(object):
	train_rle_path = 'D:/DATA/SIIM/train-rle.csv' # the path of `train-rle.csv`
	train_png_path = 'D:/DATA/SIIM/train_png'
	test_png_path = 'D:/DATA/SIIM/test_png'
	sub_sample_path = 'D:/DATA/SIIM/sample_submission.csv'
	leak_path = 'D:/DATA/SIIM/leak-cls.csv'

	init_checkpoint = 'deeplab.pth' # the path of pretrained checkpoint, used to finetuning.
	init_optimizer = 'optimizer.pth'

	model = 'unet' # 'unet'
	upsample = 'bilinear' # 'CARAFE', 'bilinear' the method of upsample

	seed = 2019
	size = 512 # image size
	num_epochs = 100
	batch_size = 8 
	num_workers = 0
	lr = 0.00001

	Gradient_Accumulation = True
	accumulation_steps = max(1, 36//batch_size)

	interval = 10000/batch_size//100 # print interval

config = DefaultConfigs()