import torch as t
from skimage import io
import os
from torchvision import transforms
from torch.utils.data import Dataset


# import fnmatch
# print len(fnmatch.filter(os.listdir(dirpath), '*.txt'))

class CustomDataset(Dataset):

    def __init__(self, root_dir, is_segment=False):

        self.is_segment=is_segment

        self.composed = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.34, 0.33, 0.35), (0.19, 0.18, 0.18))])

        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.ir_dir = os.path.join(root_dir, 'ir')

        # Check dir exist
        assert os.path.isdir(self.rgb_dir), 'Error: No rgb dir'
        assert os.path.isdir(self.ir_dir), 'Error: No ir dir'

        _, _, files1 = next(os.walk(self.rgb_dir))
        _, _, files2 = next(os.walk(self.ir_dir))

        assert len(files1) == len(files2), f"files are different rgb:{len(files1)}, ir:{len(files2)} "

        if self.is_segment:
            self.segment_dir = os.path.join(root_dir, 'segment')
            assert os.path.isdir(self.segment_dir), 'Error: No segment dir'
            _, _, files3 = next(os.walk(self.segment_dir))
            assert len(files3) == len(files2), f"files are different rgb:{len(files1)}, segment:{len(files3)}"

        self.L = len(files1)
        print(f"Custom dataset initialized with {self.L} images")

    def __len__(self):
        return self.L

    def __getitem__(self, index):

        rgb = (io.imread(os.path.join(self.rgb_dir, f'{index}.jpg'))) / 255.0
        rgb = rgb.transpose(2, 0, 1)
        rgb = self.composed(rgb)

        ir = (io.imread(os.path.join(self.ir_dir, f'{index}.jpg'))) / 255.0
        ir = ir.transpose(2, 0, 1)
        ir = self.composed(ir)

        if self.is_segment:
            segment = (io.imread(os.path.join(self.segment_dir, f'{index}.jpg'))) / 255.0
            segment = segment.transpose(2, 0, 1)
            segment = self.composed(segment)
            return [rgb, ir, segment]

        return [rgb, ir]
