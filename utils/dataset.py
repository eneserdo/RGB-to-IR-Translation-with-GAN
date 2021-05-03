import torch as t
# from torch.utils.data import DataLoader
import os

class Dataset(t.utils.data.Dataset):

    def __init__(self, root_dir):

        rgb_dir=os.path.join(root_dir, 'rgb')
        ir_dir=os.path.join(root_dir, 'ir')
        segment_dir=os.path.join(root_dir, 'segment')

        self.labels = None
        self.data=None

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        data_t=t.tensor(self.data[index], dtype=t.float)
        label_t=t.tensor(self.labels[index], dtype=t.float)

        return data_t, label_t