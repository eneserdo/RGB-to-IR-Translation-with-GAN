import torch as t
from torch.utils.data import DataLoader
import tqdm
import torch.optim as optim
from utils import parser, utils, dataset
from models import networks, losses
import os
import numpy as np


def main(opt):
    print(opt)

    # Parameters
    lambda_D = 5.0
    lambda_FM = 1
    lambda_P = 0.5

    epoch = 1000
    batch_size = 10
    nf = 1

    ####### Preparation for Training #######
    # Load the networks
    if t.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'

    print(f"Device: {device}")

    disc = networks.MultiScaleDisc(input_nc=3, ndf=nf).to(device)
    gen = networks.Generator(input_nc=3, output_nc=1, ngf=nf, n_blocks=7, transposed=opt.transposed).to(device)

    if opt.current_epoch != 0:

        disc.load_state_dict(t.load(os.path.join(opt.checkpoints_dir, f"discriminator_{opt.current_epoch}.pth")))
        gen.load_state_dict(t.load(os.path.join(opt.checkpoints_dir, f"generator_{opt.current_epoch}.pth")))

    else:

        utils.weights_init(disc)
        utils.weights_init(gen)

    loss_change_g = []
    loss_change_d = []

    # Create optimizers
    optim_g = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_d = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Create loss functions
    loss = losses.GanLoss()
    loss_fm = losses.FeatureMatchingLoss()
    loss_p = losses.VGGLoss()

    # Create dataloader
    ds = dataset.CustomDataset(opt.data_dir)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    # Start to training
    print("Training is starting...")
    i = 0
    for e in range(epoch):
        print(f"Epoch #{e}")
        for data in tqdm(dataloader):
            i += 1

            rgb = data[0].to(device)
            ir = data[1].to(device)
            if opt.segment:
                segment = data[2].to(device)

            condition = t.cat([rgb, segment], dim=1)

            ir_pred = gen(condition)

            # Updating Generator
            optim_g.zero_grad()

            out1_pred, out2_pred = disc(ir_pred)
            out1, out2 = disc(ir)

            gen_loss = loss(out1_pred[-1], out2_pred[-1], is_real=True) * lambda_D + (
                    loss_fm(out1_pred[:-1], out1[:-1]) + loss_fm(out2_pred[:-1], out2[:-1])) * lambda_FM + \
                       +lambda_P * loss_p(ir_pred, ir)

            # loss_change_g
            gen_loss.backward()

            optim_g.step()

            # Updating Discriminator.
            optim_d.zero_grad()

            disc_loss = loss(out1_pred[-1].detach(), out2_pred[-1].detach(), is_real=False) + loss(out1[-1], out2[-1],
                                                                                                   is_real=True)
            disc_loss.backward()

            optim_d.step()

            if i%100==1:
                utils.show_tensor_images(ir_pred, i, opt.results_dir, 'pred')
                utils.show_tensor_images(ir, i, opt.results_dir, 'ir')
                utils.show_tensor_images(rgb, i, opt.results_dir, 'rgb')
                print('Example images saved')


        t.save(disc.state_dict(), os.path.join(opt.checkpoints_dir, f"discriminator_{e}.pth"))
        t.save(gen.state_dict(), os.path.join(opt.checkpoints_dir, f"generator_{e}.pth"))
        print("Models saved")


if __name__ == '__main__':

    args = parser.Parser()
    opt = args.initialize()

    if not os.path.isdir(opt.checkpoints_dir):
        os.mkdir(opt.checkpoints_dir)
        print("Checkpoints directory was created")

    if not os.path.isdir(opt.results_dir):
        os.mkdir(opt.results_dir)
        print("Example directory was created")

    main(opt)
