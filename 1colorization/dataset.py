import torch.utils.data as data
import torch
import h5py
import random
import numpy as np

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path,arg=True):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.arg=arg
        self.gray = hf.get('data')
        self.label = hf.get('rgb')

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, index):
        gray = self.gray[index]
        label = self.label[index]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            gray = self.arguement(gray, rotTimes, vFlip, hFlip)
            label = self.arguement(label, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(gray), np.ascontiguousarray(label)
        
    def __len__(self):
        return self.label.shape[0]