import argparse


class Parser:
    def __init__(self):
        self.arg=argparse.ArgumentParser()

    def initialize(self):

        self.arg.add_argument('--current_epoch',default=0, help="Enter the epoch numper to continue the training")
        self.arg.add_argument('--transposed',default=False, help="Use transposed convolution")
        self.arg.add_argument('--segment',default=False, help="Use segmentation images")
        self.arg.add_argument('--loss',default='lsgan', help="Enter the loss type: lsgan or gan")
        self.arg.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.arg.add_argument('--results_dir', type=str, default='./examples', help='Results are saved here')
        self.arg.add_argument('--data_dir', type=str, default='./dataset', help='Enter the dataset directory')

        return self.arg.parse_args()


class TestParser:
    def __init__(self):
        self.arg=argparse.ArgumentParser()

    def initialize(self):

        self.arg.add_argument('--current_epoch',default=0, help="Enter the epoch numper to continue the training")
        self.arg.add_argument('--segment',default=False, help="Use segmentation images")
        self.arg.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.arg.add_argument('--out_dir', type=str, default='./test_results', help='Results are saved here')

        return self.arg.parse_args()

