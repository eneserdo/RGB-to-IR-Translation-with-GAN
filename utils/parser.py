import argparse


class Parser:
    def __init__(self):
        self.arg=argparse.ArgumentParser()

    def initialize(self):

        self.arg.add_argument('--current_epoch',default=0, help="Enter the epoch numper to continue the training")
        self.arg.add_argument('--transposed',default=False, help="Use transposed convolution")
        self.arg.add_argument('--segment',default=False, help="Use segmentation images")
        self.arg.add_argument('--loss',default='lsgan', help="Enter the loss type: lsgan or gan")
        self.arg.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
        self.arg.add_argument('--results_dir', type=str, default='examples', help='Results are saved here')
        self.arg.add_argument('--data_dir', type=str, default='dataset', help='Enter the dataset directory')
        self.arg.add_argument('--amp', type=bool, default=False, help='To use automatic mixed precision')
        self.arg.add_argument('--save_freq', type=int, default=500, help='Image saving frequency')

        return self.arg.parse_args()


class TestParser:
    def __init__(self):
        self.arg=argparse.ArgumentParser()

    def initialize(self):

        self.arg.add_argument('--current_epoch', required=True, help="Enter the epoch numper of the models")
        self.arg.add_argument('--segment',default=False, help="Use segmentation images")
        self.arg.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models directory to load')
        self.arg.add_argument('--out_dir', type=str, default='test_results', help='Results are saved here')
        self.arg.add_argument('--inp_dir', type=str, default='testset', help='Input images')

        return self.arg.parse_args()


class ResizeParser:
    def __init__(self):
        self.arg=argparse.ArgumentParser()

    def initialize(self):

        self.arg.add_argument('--out_dir', type=str, default='dataset', help='Results are saved here')
        self.arg.add_argument('--rgb_dir', type=str, required=True, help='Input images')
        self.arg.add_argument('--ir_dir', type=str, required=True, help='Input images')

        return self.arg.parse_args()

