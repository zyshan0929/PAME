import torch
import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils.utils import seed_torch



class SJTUDataset(Dataset):
    def __init__(self, mode: str, fold: int = 0):
        index_root = './data/index/SJTU-PCQA'
        modeDict = {
            'train': 'split_{}_train.xlsx'.format(str(fold)),
            'test': 'split_{}_test.xlsx'.format(str(fold)),
            'train_6view': 'split_{}_train_6view.xlsx'.format(str(fold)),
            'test_6view': 'split_{}_test_6view.xlsx'.format(str(fold)),
            'total_6view': 'total_6view.xlsx'.format(str(fold)),
            'total': 'total.xlsx'
        }
        index_file_name = modeDict[mode]
        index_file_name = os.path.join(index_root, index_file_name)
        self.file = pd.read_excel(index_file_name)
        self.mode = mode
        self.crop_size = 224
        self.npoint = 2048
        if mode == 'train':
            self.transform = transforms.Compose([
                # transforms.RandomCrop(self.crop_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
        self.transform_compose = transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file['score'])

    def __getitem__(self, idx):
        imgs = torch.zeros((6, 3, self.crop_size, self.crop_size))
        imgs_path = self.file.iloc[idx]['imgs_path']
        # name = self.file.iloc[idx]['name']
        for view in range(6):
            img_path = imgs_path + '{}.png'.format(view)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs[view] = img

        img_compose_path = self.file.iloc[idx]['compose_path']
        img_compose = Image.open(img_compose_path).convert('RGB')
        img_compose = self.transform_compose(img_compose)


        # selected_patches = torch.zeros((6, 3, self.npoint))
        # pc_path = self.file.iloc[idx]['pc_path']
        # points = list(np.load(pc_path))
        # if 'train' in self.mode:
        #     random_patches = random.sample(points, 6)
        # else:
        #     random_patches = points
        # for i in range(6):
        #     selected_patches[i] = torch.from_numpy(random_patches[i]).transpose(0,1)

        mos = -0.5 + torch.tensor(self.file.iloc[idx]['score'], dtype=torch.float32)/10
        # return imgs, mos
        return imgs, img_compose, mos
