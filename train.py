"""
    This file saves the model after every epoch. For further training of n epochs trained model,
     specify the '-e', '--current_epoch' parameters.
    If you want to use different data, do not forget to modify the utils.dataset
"""

import os
import time

import torch as t
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import networks, losses
from utils import parser, utils, dataset


def main(opt):
    # Training config
    print(opt)

    t.manual_seed(0)

    # Parameters
    lambda_FM = 10
    lambda_P = 10
    lambda_2 = opt.lambda_second

    nf = 64  # 64
    n_blocks = 6  # 6

    # Load the networks
    if t.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'

    print(f"Device: {device}")

    if opt.segment:
        disc = networks.MultiScaleDisc(input_nc=4, ndf=nf).to(device)
        gen = networks.Generator(input_nc=6, output_nc=1, ngf=nf, n_blocks=n_blocks, transposed=opt.transposed).to(device)
    else:
        disc = networks.MultiScaleDisc(input_nc=1, ndf=nf).to(device)
        gen = networks.Generator(input_nc=3, output_nc=1, ngf=nf, n_blocks=n_blocks, transposed=opt.transposed).to(device)

    if opt.current_epoch != 0:
        disc.load_state_dict(t.load(os.path.join(opt.checkpoints_file, f"e_{opt.current_epoch:0>3d}_discriminator.pth")))
        gen.load_state_dict(t.load(os.path.join(opt.checkpoints_file, f"e_{opt.current_epoch:0>3d}_generator.pth")))
        print(f"e_{opt.current_epoch:0>3d}_generator.pth was loaded")
        print(f"e_{opt.current_epoch:0>3d}_discriminator.pth was loaded")

    else:
        disc.apply(utils.weights_init)
        gen.apply(utils.weights_init)
        print("Weights are initialized")

    # Losses to track
    # # Main losses
    loss_change_g = []
    loss_change_d = []
    # # Components
    loss_change_fm1 = []
    loss_change_fm2 = []
    loss_change_d1 = []
    loss_change_d2 = []
    loss_change_g1 = []
    loss_change_g2 = []
    loss_change_p = []

    # Create optimizers (Notice the lr of discriminator)
    optim_g = optim.Adam(gen.parameters(), lr=opt.learning_rate/5, betas=(0.5, 0.999))
    optim_d = optim.Adam(disc.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)

    # Create Schedulers
    # g_scheduler = t.optim.lr_scheduler.LambdaLR(optim_g, utils.lr_lambda)
    # d_scheduler = t.optim.lr_scheduler.LambdaLR(optim_d, utils.lr_lambda)

    # Create loss functions
    loss = losses.GanLoss()
    loss_fm = losses.FeatureMatchingLoss()
    loss_p = losses.VGGLoss(device)  # perceptual loss

    # Create dataloader
    ds = dataset.CustomDataset(opt.data_dir, is_segment=opt.segment, sf=opt.scale_factor)
    dataloader = DataLoader(ds, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    # Start to training
    print("Training is starting...")
    i = 0
    for e in range(1 + opt.current_epoch, 1 + opt.training_epoch + opt.current_epoch):
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

            # # # Updating Discriminator # # #
            optim_d.zero_grad()

            out1_pred, out2_pred = disc(ir_pred.detach())   # It returns a list [fms... + output]

            l_d_pred1, l_d_pred2 = loss(out1_pred[-1], out2_pred[-1], is_real=False)
            l_d_real1, l_d_real2 = loss(out1[-1], out2[-1], is_real=True)

            l_d_scale1 = l_d_pred1 + l_d_real1
            l_d_scale2 = l_d_pred2 + l_d_real2

            disc_loss = l_d_scale1 + l_d_scale2 * lambda_2

            # Normalize the loss, and track
            loss_change_d += [disc_loss.item() / opt.batch_size]
            loss_change_d1 += [l_d_scale1.item() / opt.batch_size]
            loss_change_d2 += [l_d_scale2.item() / opt.batch_size]

            disc_loss.backward()
            optim_d.step()

            # # # Updating Generator # # #
            optim_g.zero_grad()

            out1_pred, out2_pred = disc(ir_pred)    # It returns a list [fms... + output]

            fm_scale1 = loss_fm(out1_pred[:-1], out1[:-1])
            fm_scale2 = loss_fm(out2_pred[:-1], out2[:-1])

            fm = fm_scale1 + fm_scale2 * lambda_2

            perceptual = loss_p(ir_pred, ir)

            loss_change_fm1 += [fm_scale1.item() / opt.batch_size]
            loss_change_fm2 += [fm_scale2.item() / opt.batch_size]

            loss_change_p += [perceptual.item() / opt.batch_size]

            l_g_scale1, l_g_scale2 = loss(out1_pred[-1], out2_pred[-1], is_real=True)
            gen_loss = l_g_scale1 + l_g_scale2 * lambda_2 + fm * lambda_FM + perceptual * lambda_P

            loss_change_g += [gen_loss.item() / opt.batch_size]
            loss_change_g1 += [l_g_scale1.item() / opt.batch_size]
            loss_change_g2 += [l_g_scale2.item() / opt.batch_size]

            gen_loss.backward()
            optim_g.step()

            # Save images
            if i % opt.img_save_freq == 1:
                utils.save_tensor_images(ir_pred, i, opt.results_file, 'pred')
                utils.save_tensor_images(ir, i, opt.results_file, 'ir')
                utils.save_tensor_images(rgb, i, opt.results_file, 'rgb')
                print('\nExample images saved')

                print("Losses:")
                print(f"G: {loss_change_g[-1]:.4f}; D: {loss_change_d[-1]:.4f}")
                print(f"G1: {loss_change_g1[-1]:.4f}; G2: {loss_change_g2[-1]:.4f}")
                print(f"D1: {loss_change_d1[-1]:.4f}; D2: {loss_change_d2[-1]:.4f}")
                print(f"FM1: {loss_change_fm1[-1]:.4f}; FM2: {loss_change_fm2[-1]:.4f}; P: {loss_change_p[-1]:.4f}")

        # g_scheduler.step()
        # d_scheduler.step()

        print(f"Epoch duration: {int((time.time() - start) // 60):5d}m {(time.time() - start) % 60:.1f}s")

        if i % opt.model_save_freq == 0:
            utils.save_model(disc, gen, e, opt.checkpoints_file)
    # End of training

    # Main losses are g and d, but I want to save all components separately
    utils.save_loss(d=loss_change_d, d1=loss_change_d1, d2=loss_change_d2,
                    g=loss_change_g, g1=loss_change_g1, g2=loss_change_g2,
                    fm1=loss_change_fm1, fm2=loss_change_fm2, p=loss_change_p,
                    path=opt.loss_file, e=e)

    utils.save_model(disc, gen, e, opt.checkpoints_file)

    utils.show_loss(opt.checkpoints_file)
    print("Done!")


if __name__ == '__main__':

    args = parser.Parser(__doc__)
    opt = args()

    print(f"Working directory: {os.getcwd()}")

    if not os.path.isdir(opt.checkpoints_file):
        os.mkdir(opt.checkpoints_file)
        print("checkpoints directory was created")

    if not os.path.isdir(opt.results_file):
        os.mkdir(opt.results_file)
        print("example directory was created")

    if not os.path.isdir(opt.loss_file):
        os.mkdir(opt.loss_file)
        print("tracked_losses directory was created")

    if opt.amp:
        raise Warning("AMP is not implemented yet")

    main(opt)
