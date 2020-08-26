from albumentations.augmentations.transforms import Flip, Normalize, Resize, RandomBrightnessContrast, HueSaturationValue, RandomContrast
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Compose
import cv2
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms

def train_transfrom(image, size=256):
    transform = Compose([
        Resize(size,size, interpolation=cv2.INTER_AREA),
        Flip(),
        RandomBrightnessContrast(),
        HueSaturationValue(),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ToTensorV2()
        ])
    image_transform = transform(image = image)['image']

    return image_transform

def test_transfrom(image, size=256): 
    transform = Compose([ 
        Resize(size,size, interpolation=cv2.INTER_AREA),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image_transform = transform(image = image)['image']

    return image_transform

if __name__ == '__main__':
    img = cv2.imread("train/clahe/10/good/844877ff6b84b43442e2b6c7b179b40a.png", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_trans = train_transfrom(img)
    print(img_trans.shape)
    img_trans = img_trans.numpy().transpose(1,2,0)
    
    plt.imshow(img_trans)
    plt.show()
