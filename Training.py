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
    exit()
    # Parameters
    lambda_D = 5.0
    lambda_FM = 1
    lambda_P = 0.5

    epoch = 1000
    batch_size=10

    # Preparation for Training

    assert t.cuda.is_available()
    device = "cuda"

    disc=networks.MultiScaleDisc()
    gen=networks.Generator()

    if opt.current_epoch!=0:
        loss_change_g=np.load("checkpoints/loss_change_g.npy")
        loss_change_d=np.load("checkpoints/loss_change_d.npy")
        # TODO load models
    else:
        loss_change_g=np.array([])
        loss_change_d=np.array([])
        utils.weights_init(disc.disc1)
        utils.weights_init(disc.disc2)
        utils.weights_init(gen)

    optim_g=optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_d=optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))


    loss=losses.GanLoss()
    loss_fm=losses.FeatureMatchingLoss()
    loss_p=losses.VGGLoss()

    DS=dataset.Dataset(opt.data_dir)

    dataloader = DataLoader(DS, batch_size=batch_size, shuffle=True, num_workers=2)

    # Start to training

    for e in range(epoch):
        for ir, rgb, segment in tqdm(dataloader):

            condition=t.cat([rgb, segment], dim=1)

            ir_pred = gen(condition)

            # Updating Generator
            optim_g.zero_grad()

            out1_pred, out2_pred=disc(ir_pred)
            out1, out2=disc(ir)

            gen_loss=loss(out1_pred[-1], out2_pred[-1], is_real=True)*lambda_D+(loss_fm(out1_pred[:-1], out1[:-1])+loss_fm(out2_pred[:-1], out2[:-1]))*lambda_FM+ \
                +lambda_P*loss_p(ir_pred, ir)

            # loss_change_g
            gen_loss.backward()

            optim_g.step()

            # Updating Discriminator.
            optim_d.zero_grad()

            disc_loss=loss(out1_pred[-1].detach(), out2_pred[-1].detach(), is_real=False)+loss(out1[-1], out2[-1], is_real=True)

            disc_loss.backward()

            optim_d.step()



        # TODO save the model and image


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
