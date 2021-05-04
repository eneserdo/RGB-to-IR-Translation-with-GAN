import torch as t
from torch.utils.data import DataLoader
import tqdm
import torch.optim as optim
from utils import parser, utils, dataset
from models import networks, losses
import os
import numpy as np

def main(opt):
    """ TODO
    Only generator will be runned
    If there is a input image just translate that
    If there is a input folder translate these images to new folder
    """

    ####### Preparation for Testing #######
    # Load the networks
    if t.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'

    print(f"Device: {device}")

    gen = networks.Generator(input_nc=3, output_nc=1, ngf=nf, n_blocks=7, transposed=opt.transposed).to(device)

    if opt.current_epoch != 0:
        gen.load_state_dict(t.load(os.path.join(opt.checkpoints_dir, f"generator_{opt.current_epoch}.pth")))

    gen.eval()

    dataloader=None

    for img in dataloader:

        with t.no_grad():

            pred=gen()

        # TODO save images


if __name__ == '__main__':

    raise NotImplemented

    # args = parser.TestParser()
    # opt = args.initialize()
    # main(opt)
