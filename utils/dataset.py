import cv2
import os

import numpy as np
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


# import fnmatch
# print len(fnmatch.filter(os.listdir(dirpath), '*.txt'))

class CustomDataset(Dataset):

    def __init__(self, root_dir, is_segment=False, sf=1):

        self.is_segment = is_segment

        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.ir_dir = os.path.join(root_dir, 'ir')

        # Check dir exist
        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError('Error: No rgb directory')

        if not os.path.isdir(self.ir_dir):
            raise FileNotFoundError('Error: No ir directory')

        _, _, files1 = next(os.walk(self.rgb_dir))
        _, _, files2 = next(os.walk(self.ir_dir))

        assert len(files1) == len(files2), f"files are different rgb:{len(files1)}, ir:{len(files2)} "

        if self.is_segment:
            self.segment_dir = os.path.join(root_dir, 'segment')
            assert os.path.isdir(self.segment_dir), 'Error: No segment directory'
            _, _, files3 = next(os.walk(self.segment_dir))
            assert len(files3) == len(files2), f"files are different rgb:{len(files1)}, segment:{len(files3)}"

            self.composed_segment = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            print("Segment file is loaded")

        self.Length = len(files1)

        self.sf = sf

        self.composed_rgb = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.composed_ir = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(0.5, 0.5)])

        print(f"Custom dataset initialized with {self.Length} images")
        print(f"Scaling factor: {self.sf} ")

    def __len__(self):
        return self.Length

    def __getitem__(self, index):
        rgb = (io.imread(os.path.join(self.rgb_dir, f"IMG_{index:0>4d}.png"))) / 255.0
        rgb = cv2.resize(rgb, (0, 0), fx=self.sf, fy=self.sf)
        rgb = self.composed_rgb(rgb)

        ir = np.load(os.path.join(self.ir_dir, f"IMG_{index:0>4d}.npy"))
        ir = (ir - np.min(ir)) / (np.max(ir) - np.min(ir))
        ir = cv2.resize(ir, (0, 0), fx=self.sf, fy=self.sf)
        ir = self.composed_ir(ir)

        if self.is_segment:
            segment = (io.imread(os.path.join(self.segment_dir, f"IMG_{index:0>4d}.png"))) / 255.0
            segment = cv2.resize(segment, (0, 0), fx=self.sf, fy=self.sf)
            segment = self.composed_segment(segment)

            return [rgb.float(), ir.float(), segment.float()]

        return [rgb.float(), ir.float()]


# FIXME
class TestDataset(Dataset):
    def __init__(self, root_dir, is_segment=False):

        self.is_segment = is_segment

        self.composed_rgb = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.34, 0.33, 0.35), (0.19, 0.18, 0.18))])

        self.rgb_dir = os.path.join(root_dir, 'rgb')

        # Check dir exist
        assert os.path.isdir(self.rgb_dir), 'Error: No rgb dir'

        _, _, files1 = next(os.walk(self.rgb_dir))

        if self.is_segment:
            self.segment_dir = os.path.join(root_dir, 'segment')
            assert os.path.isdir(self.segment_dir), 'Error: No segment dir'
            _, _, files3 = next(os.walk(self.segment_dir))
            assert len(files3) == len(files1), f"files are different rgb:{len(files1)}, segment:{len(files3)}"

        self.L = len(files1)
        print(f"{self.L} images found")

    def __len__(self):
        return self.L

    def __getitem__(self, index):
        rgb = (io.imread(os.path.join(self.rgb_dir, f"/IMG_{i:0>4d}.png"))) / 255.0
        # rgb = (io.imread(os.path.join(self.rgb_dir, f'{index}.jpg'))) / 255.0
        rgb = self.composed_rgb(rgb)

        if self.is_segment:
            segment = (io.imread(os.path.join(self.segment_dir, f"/IMG_{i:0>4d}.png"))) / 255.0
            segment = cv2.resize(segment, (0, 0), fx=self.sf, fy=self.sf)
            segment = self.composed_segment(segment)

            return [rgb.float(), segment.float()]

        return rgb.float()
