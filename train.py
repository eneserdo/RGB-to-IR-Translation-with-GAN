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

    t.manual_seed(0)

    # Parameters
    lambda_D = 1
    lambda_FM = 10
    lambda_P = 10

    nf = 64     #64
    n_blocks = 6    #6

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

    # Create Schedulers
    g_scheduler = t.optim.lr_scheduler.LambdaLR(optim_g, utils.lr_lambda)
    d_scheduler = t.optim.lr_scheduler.LambdaLR(optim_d, utils.lr_lambda)

    # Create loss functions
    loss = losses.GanLoss()
    loss_fm = losses.FeatureMatchingLoss()
    loss_p = losses.VGGLoss(device)  # perceptual loss

    # Create dataloader
    ds = dataset.CustomDataset(opt.data_dir, is_segment=opt.segment, sf=opt.scale_factor)
    dataloader = DataLoader(ds, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    # t.autograd.set_detect_anomaly(True)

    # Start to training
    print("Training is starting...")
    i = 0
    for e in range(1+opt.current_epoch, 1 + opt.training_epoch + opt.current_epoch):
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
            loss_change_d += [disc_loss.item() / opt.batch_size]

            disc_loss.backward()
            optim_d.step()

            # Updating Generator
            optim_g.zero_grad()

            out1_pred, out2_pred = disc(ir_pred)

            fm = loss_fm(out1_pred[:-1], out1[:-1]) + loss_fm(out2_pred[:-1], out2[:-1])
            perceptual = loss_p(ir_pred, ir)

            gen_loss = loss(out1_pred[-1], out2_pred[-1], is_real=True) * lambda_D + \
                       fm * lambda_FM + perceptual * lambda_P

            loss_change_g += [gen_loss.item() / opt.batch_size]

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
                    f"FM: {fm.item() / opt.batch_size:.4f}; P: {perceptual.item() / opt.batch_size:.4f}; G: {loss_change_g[-1]:.4f}; D: {loss_change_d[-1]:.4f}")

        g_scheduler.step()
        d_scheduler.step()
        print(optim_g)

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
