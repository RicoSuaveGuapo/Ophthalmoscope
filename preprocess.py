import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm
from shutil import copyfile, move
import time

from torch.utils.data import Dataset
from PIL import Image, ImageCms
import os
import torchvision.transforms as transforms

sys.path.append('/home/rico-li/Job/Ophthalmoscope/EyeQ/EyeQ/MCF_Net')
sys.path.append('/home/rico-li/Job/Ophthalmoscope/EyeQ/EyeQ/')
sys.path.append('/home/rico-li/Job/Ophthalmoscope/EyeQ/EyeQ/EyeQ_preprocess')
from networks.densenet_mcf import dense121_mcs
import torchvision.transforms as transforms
from networks.densenet_mcf import dense121_mcs
from EyeQ_loader import DatasetGenerator

from EyeQ_process_main import process
from albumentations.pytorch.transforms import ToTensorV2


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

if __name__ == "__main__":
    mode = 'test'
    assert mode in ['train','val','test']
    # move file (one time only)
    # img_ori_path = f'{mode}/original'
    # if not os.path.exists(img_ori_path):
    #     os.mkdir(img_ori_path)
    # else:
    #     print('original image directory exist')

    # path = f'{mode}'
    # image_name = os.listdir(path)
    # image_path = [os.path.join(path, image) for image in image_name if image[-4:] == '.jpg']
    # copy_action = [move(img_path, img_ori_path) for img_path in image_path]

    # quality model preprocess
    clahe = 'clahe'
    assert clahe in ['clahe', 'no_clahe']

    # save_path = f'{mode}/processed'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # else:
    #     print('process directory exist')

    # img_path = f'{mode}/original'
    # image_list = os.listdir(img_path)
    # image_list = [image for image in image_list if image[-4:] == '.jpg']
    # start_time = time.time()
    # process(image_list, save_path, img_path)
    # print(f'processed time: %.2f' % (time.time()-start_time))

    loaded_model = torch.load('/home/rico-li/Job/Ophthalmoscope/EyeQ/EyeQ/DenseNet121_v3_v1.tar')
    model = dense121_mcs(n_class=3)
    model.load_state_dict(loaded_model['state_dict'])
    model.eval()
    print('model ready')

    df = pd.read_csv('ophthalmoscope_v3.csv')
    print('Dataframe ready')

    data_dir = f'{mode}/processed'

    transform1 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])
    transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    datasets = DatasetGenerator(data_dir=data_dir, transform1=transform1,
                            transform2=transform2, n_class=3, set_name='test', clahe=clahe)

    dataloader = torch.utils.data.DataLoader(datasets, batch_size = 1, shuffle=False, num_workers=0, pin_memory=True)

    print('dataset size:', len(datasets))
    print('dataset ready')

    if not os.path.exists(f'{mode}/{clahe}'):
        print(f'{mode} clahe dir does not exist')
        os.mkdir(f'{mode}/{clahe}/')
        for i in range(1, 11):
            os.mkdir(f'{mode}/{clahe}/{i}0')
            os.mkdir(f'{mode}/{clahe}/{i}0/good')
            os.mkdir(f'{mode}/{clahe}/{i}0/useable')
            os.mkdir(f'{mode}/{clahe}/{i}0/reject')
            print(f'mk {mode}/{clahe}/{i}0 file done')

    if torch.cuda.is_available():
        model.cuda()
    good = 0
    useable = 0
    reject = 0
    if 'quality' in df.columns:
        pass
    else:
        df['quality'] = np.nan

    # quality: 0:good, 1:useable, 2:reject, Nan: if no image
    for imagesA, imagesB, imagesC, name in tqdm(dataloader, desc='Process'):
        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()

        begin_time = time.time()
        _, _, _, _, result_mcs = model(imagesA, imagesB, imagesC)

        index = result_mcs.argmax()
        file_name = name[0].split('/')[-1]
        df_small = df[df['path'] == file_name[:-4]]
        idx = df[df['path'] == file_name[:-4]].index.values
        age_class = (int(df_small['age'].values -1) //10 + 1) * 10

        if clahe == 'clahe':
            # image = imagesA.cpu().squeeze().numpy().transpose(1,2,0) # suitable for densenet model
            image = cv2.imread(name[0])
            clahe_fun = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_r = clahe_fun.apply(image[...,0])
            img_g = clahe_fun.apply(image[...,1])
            img_b = clahe_fun.apply(image[...,2])
            img_r = np.expand_dims(img_r, axis=2)
            img_g = np.expand_dims(img_g, axis=2)
            img_b = np.expand_dims(img_b, axis=2)
            image = np.concatenate((img_r,img_g,img_b), axis=2)

            if index.item() == 0:
                dst_path = os.path.join(f'{mode}', f'{clahe}',f'{age_class}', 'good', f'{file_name}')
                cv2.imwrite(dst_path, image)
                df.loc[idx,'quality'] = index.item()
                good += 1
            elif index.item() == 1:
                dst_path = os.path.join(f'{mode}', f'{clahe}', f'{age_class}', 'useable', f'{file_name}')
                cv2.imwrite(dst_path, image)
                df.loc[idx,'quality'] = index.item()
                useable += 1
            elif index.item() == 2:
                dst_path = os.path.join(f'{mode}', f'{clahe}', f'{age_class}', 'reject', f'{file_name}')
                cv2.imwrite(dst_path, image)
                df.loc[idx,'quality'] = index.item()
                reject += 1

        else:
            if index.item() == 0:
                copyfile(name[0], os.path.join(f'{mode}', f'{clahe}',f'{age_class}', 'good', f'{file_name}'))
                good += 1
            elif index.item() == 1:
                copyfile(name[0], os.path.join(f'{mode}', f'{clahe}', f'{age_class}', 'useable', f'{file_name}'))
                useable += 1
            elif index.item() == 2:
                copyfile(name[0], os.path.join(f'{mode}', f'{clahe}', f'{age_class}', 'reject', f'{file_name}'))
                reject += 1
    df.to_csv('ophthalmoscope_v3.csv', index=False)
        
    print(f'\nGood count {good}')
    print(f'Useable count {useable}')
    print(f'Reject count {reject}')
    # 6cd822ced6b72b58ea6e9e55925a6138.png reject with no clshe