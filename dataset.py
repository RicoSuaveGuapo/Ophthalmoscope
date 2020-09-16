import os
import time
from sys import platform
from matplotlib import pyplot as plt

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transform import train_transfrom, test_transfrom

class FundusDataset(Dataset):

    def __init__(self, mode='train', clahe='clahe', quality='good',transform = None, image_size = 256, seed = 42):
        super().__init__()

        if os.getcwd() == '/home/rico-li/Job/Ophthalmoscope':
            print('In local machine.')
        else:
            print('In server.')
        
        assert mode in ['train', 'test', 'val']
        assert quality in ['good', 'useable','reject']
        assert clahe in ['clahe', 'no_clahe']
        paths = f'{mode}'
        paths = os.path.join(paths,'clahe')
        paths = [os.path.join(paths, class_name) for class_name in os.listdir(paths)]
        paths = [os.path.join(path, f'{quality}') for path in paths]
        # e.g. ['train/clahe/10/good','train/clahe/20/good'...]
        image_paths = [os.path.join(path, file_name) for path in paths for file_name in os.listdir(path)] 
        # e.g. ['train/clahe/10/good/844877ff6b84b43442e2b6c7b179b40a.png',..]
        if platform == 'linux':
            # combine age, e.g. 10 and 20 -> =<20 class
            labels = [(int(int(image_path.split('/')[2])/10)-1)//2 for image_path in image_paths]
        elif platform == 'win32':
            labels = [(int(int(image_path.split('\\')[2])/10)-1)//2 for image_path in image_paths]
        else:
            print('Non-support OS')
            raise OSError

        self.image_paths = image_paths
        self.labels = labels
        self.mode = mode
        self.transform = transform
        self.image_size = image_size     
        np.random.seed(seed)  
            
    def __len__(self):
        return len(self.image_paths)

    def classWeight(self):
        w_class = []
        for i in range(5):
            class_i = [j for j in self.labels if j == i]
            w_class += [self.__len__()/len(class_i)]
        return w_class

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx])
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = train_transfrom(image, size=self.image_size)
        else:
            image = test_transfrom(image, size=self.image_size)

        return image, label


if __name__ == '__main__':
    from utils import UnNormalize
    mode = 'train'
    train_dataset = FundusDataset(mode=mode, transform=True, image_size=256)
    # print(train_dataset.classWeight())
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    age = torch.Tensor().type(torch.long)
    for image, label in train_dataloader:
        # image = image.view(-1,256,256)
        # unNormalize = UnNormalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        # image = unNormalize(image)
        # image = image.numpy().transpose(1,2,0)
        # plt.imshow(image)
        # plt.show()
        age = torch.cat([age, label], 0)
    print(age)
