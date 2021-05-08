import argparse
import dataset

# Todo: Add option to disable fm loss


class Parser:
    def __init__(self, des):
        self.arg=argparse.ArgumentParser(description=des)

    def __call__(self):

        self.arg.add_argument('-ce', '--current_epoch',default=0, type=int,  help="Enter the epoch numper to continue the training")
        self.arg.add_argument('--transposed',default=False, help="Use transposed convolution")
        self.arg.add_argument('--segment',default=False, help="Use segmentation images")
        self.arg.add_argument('--loss',default='lsgan', help="Enter the loss type: lsgan or gan")
        self.arg.add_argument('--checkpoints_file', type=str, default='checkpoints', help='Models are saved here')
        self.arg.add_argument('--results_file', type=str, default='examples', help='Results are saved here')
        self.arg.add_argument('-i', '--data_dir', type=str, default='dataset', help='Enter the dataset directory')
        self.arg.add_argument('--amp', type=bool, default=False, help='To use automatic mixed precision')
        self.arg.add_argument('-isf', '--img_save_freq', type=int, default=500, help='Image saving frequency')
        self.arg.add_argument('-msf', '--model_save_freq', type=int, default=3, help='Model saving frequency')
        self.arg.add_argument('-bs', '--batch_size', type=int, default=5, help='Batch size')
        self.arg.add_argument('-te', '--training_epoch', type=int, default=10, help='Number of epochs to train')
        self.arg.add_argument('-sf', '--scale_factor', type=float, default=1., help='To scale the training images')
        self.arg.add_argument('-ls', '--lambda_second', type=float, default=1., help='Lambda of downsampled image loss')

        return self.arg.parse_args()


class TestParser:
    def __init__(self):
        self.arg=argparse.ArgumentParser()

    def __call__(self):

        self.arg.add_argument('-ce', '--current_epoch', type=int, required=True, help="Enter the epoch numper of the models")
        self.arg.add_argument('--segment',default=False, help="Use segmentation images")
        self.arg.add_argument('--checkpoints_file', type=str, default='checkpoints', help='models directory to load')
        self.arg.add_argument('-o', '--out_file', type=str, default='test_results', help='Results are saved here')
        self.arg.add_argument('-i', '--inp_file', type=str, default='testset', help='Input images')
        self.arg.add_argument('-sf', '--scale_factor', type=float, default=1, help='To scale the training images')
        self.arg.add_argument('-bs', '--batch_size', type=int, default=1, help='Batch size')
        self.arg.add_argument('--transposed',default=False, help="Use transposed convolution")


        return self.arg.parse_args()


class ResizeParser:
    def __init__(self, des):
        self.arg=argparse.ArgumentParser(description=des)

    def __call__(self):

        self.arg.add_argument('-o','--out_dir', type=str, default='dataset', help='Results are saved here')
        self.arg.add_argument('-rgb','--rgb_dir', type=str, required=True, help='Input images')
        self.arg.add_argument('-ir','--ir_dir', type=str, required=True, help='Input images')

        return self.arg.parse_args()

