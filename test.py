# TODO: resize the input images

import os

import torch as t
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import networks
from utils import parser, utils, dataset


def main(opt):

    ngf = 64
    n_blocks = 6

    # Load the networks
    if t.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'

    print(f"Device: {device}")

    gen = networks.Generator(input_nc=3, output_nc=1, ngf=ngf, n_blocks=n_blocks, transposed=opt.transposed).to(device)

    gen.load_state_dict(t.load(os.path.join(opt.checkpoints_file, f"e_{opt.current_epoch}_generator.pth")))

    gen.eval()

    try:
        ds = dataset.CustomDataset(root_dir=opt.inp_file, sf=opt.scale_factor)
    except:
        print("IR images not found but it is ok")
        ds = dataset.TestDataset(root_dir=opt.inp_file)

    dataloader = DataLoader(ds, batch_size=opt.batch_size, shuffle=False, num_workers=2)

    i = 0

    for data in tqdm(dataloader):
        i += 1
        with t.no_grad():

            rgb = data[0].to(device)
            ir = data[1]


            if opt.segment:
                segment = data[-1].to(device)
                condition = t.cat([rgb, segment], dim=1)
            else:
                condition = rgb

            ir_pred = gen(condition)
            utils.save_all_images(rgb, ir, ir_pred, i, opt.out_file, resize_factor=1.)

    print("Done!")


if __name__ == '__main__':

    args = parser.TestParser()
    opt = args()

    print(f"Working directory: {os.getcwd()}")

    if not os.path.isdir(opt.out_file):
        os.mkdir(opt.out_file)
        print("Checkpoints directory was created")


    main(opt)

