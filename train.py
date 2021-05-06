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

    # Set the directories
    checkpoints_dir=os.path.join(os.getcwd(),opt.checkpoints_dir)
    results_dir=os.path.join(os.getcwd(),opt.results_dir)
    data_dir=os.path.join(os.getcwd(),opt.data_dir)

    # t.manual_seed(0)

    # Parameters
    lambda_D = 5.0
    lambda_FM = 1
    lambda_P = 0.5

    epoch = 1000
    batch_size = 2
    nf = 10
    n_blocks = 2

    # Load the networks
    if t.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'

    print(f"Device: {device}")

    disc = networks.MultiScaleDisc(input_nc=1, ndf=nf).to(device)
    gen = networks.Generator(input_nc=3, output_nc=1, ngf=nf, n_blocks=n_blocks, transposed=opt.transposed).to(device)

    if opt.current_epoch != 0:

        disc.load_state_dict(t.load(os.path.join(checkpoints_dir, f"discriminator_{opt.current_epoch}.pth")))
        gen.load_state_dict(t.load(os.path.join(checkpoints_dir, f"generator_{opt.current_epoch}.pth")))

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
    loss_p = losses.VGGLoss()

    # Create dataloader
    ds = dataset.CustomDataset(data_dir, is_segment=opt.segment)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    # Start to training
    print("Training is starting...")
    t.autograd.set_detect_anomaly(True)
    i = 0
    for e in range(epoch):
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
            out1_pred, out2_pred = disc(ir_pred.detach())

            optim_d.zero_grad()

            disc_loss = (loss(out1_pred[-1], out2_pred[-1], is_real=False) + loss(out1[-1], out2[-1], is_real=True)) * lambda_D
            loss_change_d += [disc_loss.item()/batch_size]

            disc_loss.backward()
            optim_d.step()

            # Updating Generator
            optim_g.zero_grad()

            out1_pred, out2_pred = disc(ir_pred)

            gen_loss = loss(out1_pred[-1], out2_pred[-1], is_real=True) * lambda_D + \
                       (loss_fm(out1_pred[:-1], out1[:-1]) + loss_fm(out2_pred[:-1], out2[:-1])) * lambda_FM + \
                       loss_p(ir_pred, ir) * lambda_P

            loss_change_g += [gen_loss.item()/batch_size]

            gen_loss.backward()
            optim_g.step()

            if i % opt.save_freq == 1:
                utils.show_tensor_images(ir_pred, i, results_dir, 'pred')
                utils.show_tensor_images(ir, i, results_dir, 'ir')
                utils.show_tensor_images(rgb, i, results_dir, 'rgb')
                print('Example images saved')

        print(f"Epoch duration: {int((time.time()-start)//60):5d}m {(time.time()-start)%60:.1f}s")

        utils.save_model(disc, gen, e, checkpoints_dir)


if __name__ == '__main__':

    args = parser.Parser()
    opt = args.initialize()

    print(f"Working directory: {os.getcwd()}")

    if not os.path.isdir(os.path.join(os.getcwd(),opt.checkpoints_dir)):
        os.mkdir(os.path.join(os.getcwd(),opt.checkpoints_dir))
        print("Checkpoints directory was created")

    if not os.path.isdir(os.path.join(os.getcwd(),opt.results_dir)):
        os.mkdir(os.path.join(os.getcwd(),opt.results_dir))
        print("Example directory was created")

    if opt.amp:
        raise Warning("AMP is not implemented yet")

    main(opt)

    # TODO
    # loss incelenecek
    # grad update incele
