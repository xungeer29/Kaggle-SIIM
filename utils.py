import numpy as np
import random
import os
import torch

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')


def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    rle = rle.strip()
    if rle != '-1':
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position+lengths[index]] = 1
            current_position += lengths[index]

    return mask.reshape(width, height)

def mask_to_rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1
    # https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/98317
    if lastColor == 255:
        rle.append(runStart)
        rle.append(runLength)
    return " " + " ".join(rle)

def seed_everything(seed=73):
    '''
      Make PyTorch deterministic.
    '''    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

# draw curve
def draw_curve(y, name='loss', loc='upper right'):
    figs = plt.figure()
    fig1 = figs.add_subplot(1, 1, 1)

    x = [i for i in range(len(y))]
    fig1.plot(x, y, label=name)
    fig1.legend(loc=loc)

    plt.savefig(f'./figs/{name}.jpg')
    plt.show()

if __name__ == '__main__':
    # test seed_everything
    seed = 100
    seed_everything(seed)

    # test draw_curve
    data = [i**1.5 for i in range(100)]
    draw_curve(data, name='y=x^1.5', loc='upper left')