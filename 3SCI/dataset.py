import torch.utils.data as data
import torch
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        # self.data = hf.get('snap_inverse')
        self.target = hf.get('label')
        # self.mask = hf.get('mask')
    def __getitem__(self, index):
        return torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        return self.target.shape[0]


class TestFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(TestFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.target[index, :, :, :]).float()

    def __len__(self):
        return self.target.shape[0]