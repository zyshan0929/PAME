import torch
import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class LSDataset_pretrain(Dataset):
    def __init__(self, mode: str, input_size=512, crop_size=None, num_view=6):
        index_root = './data/index/LS-PCQA'
        modeDict = {
            'pretrain': 'total_{}view_{}_np.csv'.format(str(num_view),str(input_size))
        }
        index_file_name = modeDict[mode]
        index_file_path = os.path.join(index_root, index_file_name)
        self.file = pd.read_csv(index_file_path)
        self.mode = mode
        if crop_size is not None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(crop_size),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
            self.final_input_size = crop_size
        else:
            self.transform = transforms.Compose([
                # transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
            self.final_input_size = input_size

    def __len__(self):
        return len(self.file['score'])

    def __getitem__(self, idx):
        img_np_path = self.file.iloc[idx]['img_path']
        ref_img_np_path = self.file.iloc[idx]['ref_img_path']
        img, ref_img = np.load(img_np_path), np.load(ref_img_np_path)
        # img, ref_img = Image.open(img_np_path), Image.open(ref_img_np_path)
        img, ref_img = self.transform(img), self.transform(ref_img)

        mos = torch.tensor(self.file.iloc[idx]['score'], dtype=torch.float32)/5
        return img, ref_img, mos


class LSDataset(Dataset):
    def __init__(self, mode: str, input_size=512, crop_size=None, num_view=6):
        index_root = './data/index/LS-PCQA'
        modeDict = {
            'train': 'train_{}view_{}.csv'.format(str(num_view),str(input_size)),
            'test': 'test_{}view_{}.csv'.format(str(num_view),str(input_size)),
            'total': 'total.xlsx'
        }
        index_file_name = modeDict[mode]
        index_file_path = os.path.join(index_root, index_file_name)
        self.file = pd.read_csv(index_file_path)
        self.mode = mode
        self.crop_size = crop_size
        self.num_view = num_view
        if crop_size is not None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(crop_size),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
            self.final_input_size = crop_size
        else:
            self.transform = transforms.Compose([
                # transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
            self.final_input_size = input_size

    def __len__(self):
        return len(self.file['score'])//self.num_view

    def __getitem__(self, idx):
        imgs_path = list(self.file.loc[idx*self.num_view:(idx+1)*self.num_view-1,'img_path'])
        imgs = torch.zeros((self.num_view, 3, self.crop_size, self.crop_size))
        for view in range(self.num_view):
            img = Image.open(imgs_path[view]).convert('RGB')
            img = self.transform(img)
            imgs[view,...] = img
        mos = torch.tensor(self.file.iloc[idx]['score'], dtype=torch.float32)/5
        return imgs, mos