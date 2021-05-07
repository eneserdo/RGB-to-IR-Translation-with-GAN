import torch as t
from torch.utils.data import DataLoader
import tqdm
import torch.optim as optim
from utils import parser, utils, dataset
from models import networks, losses
import os
import numpy as np


def main(opt):

    # Load the networks
    if t.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'

    print(f"Device: {device}")

    gen = networks.Generator(input_nc=3, output_nc=1, ngf=64, n_blocks=7, transposed=opt.transposed).to(device)

    if opt.current_epoch != 0:
        gen.load_state_dict(t.load(os.path.join(opt.checkpoints_dir, f"generator_{opt.current_epoch}.pth")))

    gen.eval()

    try:
        dataloader = dataset.CustomDataset()
    except:
        print("IR images not found but it is ok")
        dataloader = dataset.TestDataset()

    i = 0

    for data in dataloader:
        i += 1
        with t.no_grad():

            rgb = data[0].to(device)

            if opt.segment:
                segment = data[-1].to(device)
                condition = t.cat([rgb, segment], dim=1)
            else:
                condition = rgb

            ir_pred = gen(condition)
            utils.save_tensor_images(ir_pred, i, opt.out_dir, 'pred')


if __name__ == '__main__':

    args = parser.TestParser()
    opt = args()
    main(opt)

