"""
    This file saves the model after every epoch. For further training of n epochs trained model,
     specify the '-e', '--current_epoch' parameters.
    If you want to use different data, do not forget to modify the utils.dataset
"""

import torch as t
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.optim as optim
from utils import parser, utils, dataset
from models import networks, losses
import os, time
import numpy as np


def main(opt):
    # Training config
    print(opt)

    # # Set the directories
    # checkpoints_dir=opt.checkpoints_dir
    # results_dir=opt.results_dir
    # data_dir=opt.data_dir

    # t.manual_seed(0)

    # Parameters
    lambda_D = 5.0
    lambda_FM = 1
    lambda_P = 0.5

    epoch = 10
    batch_size = 5
    nf = 64
    n_blocks = 7

    # Load the networks
    if t.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'

    print(f"Device: {device}")

    # TODO: if segment exists modify input_nc
    disc = networks.MultiScaleDisc(input_nc=1, ndf=nf).to(device)
    gen = networks.Generator(input_nc=3, output_nc=1, ngf=nf, n_blocks=n_blocks, transposed=opt.transposed).to(device)

    if opt.current_epoch != 0:
        disc.load_state_dict(t.load(os.path.join(opt.checkpoints_file, f"discriminator_{opt.current_epoch}.pth")))
        gen.load_state_dict(t.load(os.path.join(opt.checkpoints_file, f"generator_{opt.current_epoch}.pth")))

    else:
        disc.apply(utils.weights_init)
        gen.apply(utils.weights_init)

    loss_change_g = []
    loss_change_d = []

    # Create optimizers
    optim_g = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_d = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Create loss functions
    loss = losses.GanLoss()
    loss_fm = losses.FeatureMatchingLoss()
    loss_p = losses.VGGLoss()  # perceptual loss

    # Create dataloader
    ds = dataset.CustomDataset(opt.data_dir, is_segment=opt.segment)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    # t.autograd.set_detect_anomaly(True)

    # Start to training
    print("Training is starting...")
    i = 0
    for e in range(1+opt.current_epoch, 1 + epoch + opt.current_epoch):
        print(f"---- Epoch #{e} ----")
        start = time.time()

        for data in tqdm(dataloader):
            i += 1

            rgb = data[0].to(device)
            ir = data[1].to(device)

            if opt.segment:
                segment = data[2].to(device)
                condition = t.cat([rgb, segment], dim=1)
            else:
                condition = rgb

            out1, out2 = disc(ir)
            ir_pred = gen(condition)

            # Updating Discriminator
            optim_d.zero_grad()

            out1_pred, out2_pred = disc(ir_pred.detach())

            disc_loss = (loss(out1_pred[-1], out2_pred[-1], is_real=False) + loss(out1[-1], out2[-1],
                                                                                  is_real=True)) * lambda_D
            loss_change_d += [disc_loss.item() / batch_size]

            disc_loss.backward()
            optim_d.step()

            # Updating Generator
            optim_g.zero_grad()

            out1_pred, out2_pred = disc(ir_pred)

            fm = loss_fm(out1_pred[:-1], out1[:-1]) + loss_fm(out2_pred[:-1], out2[:-1])
            perceptual = loss_p(ir_pred, ir)

            gen_loss = loss(out1_pred[-1], out2_pred[-1], is_real=True) * lambda_D + \
                       fm * lambda_FM + perceptual * lambda_P

            loss_change_g += [gen_loss.item() / batch_size]

            gen_loss.backward()
            optim_g.step()

            # Save images
            if i % opt.img_save_freq == 1:
                utils.save_tensor_images(ir_pred, i, opt.results_file, 'pred')
                utils.save_tensor_images(ir, i, opt.results_file, 'ir')
                utils.save_tensor_images(rgb, i, opt.results_file, 'rgb')
                print('Example images saved')

                print("Losses:")
                print(
                    f"FM: {fm.item() / batch_size:.2f}; P: {perceptual.item() / batch_size:.2f}; G: {loss_change_g[-1]:.2f}; D: {loss_change_d[-1]:.2f}")

        print(f"Epoch duration: {int((time.time() - start) // 60):5d}m {(time.time() - start) % 60:.1f}s")

        if i % opt.model_save_freq == 0:
            utils.save_model(disc, gen, e, opt.checkpoints_file)

    np.save(os.path.join(opt.checkpoints_file, f'd_loss_v{e}.npy'), np.array(loss_change_d))
    np.save(os.path.join(opt.checkpoints_file, f'g_loss_v{e}.npy'), np.array(loss_change_g))

    utils.save_model(disc, gen, e, opt.checkpoints_file)

    utils.show_loss(opt.checkpoints_file)
    print("Done!")


if __name__ == '__main__':

    args = parser.Parser(__doc__)
    opt = args()

    print(f"Working directory: {os.getcwd()}")

    if not os.path.isdir(opt.checkpoints_file):
        os.mkdir(opt.checkpoints_file)
        print("Checkpoints directory was created")

    if not os.path.isdir(opt.results_file):
        os.mkdir(opt.results_file)
        print("Example directory was created")

    if opt.amp:
        raise Warning("AMP is not implemented yet")

    main(opt)
