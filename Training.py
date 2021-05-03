import torch as t
import tqdm
import torch.optim as optim
from utils import parser, utils
from models import networks, losses


def main():

    # Parameters
    lambda_D = 1
    lambda_FM = 1
    lambda_P = 1

    epoch = 1000

    # Preparation for Training

    assert t.cuda.is_available()
    device = "cuda"

    disc=networks.MultiScaleDisc()
    gen=networks.Generator()

    utils.weights_init(disc.disc1)
    utils.weights_init(disc.disc2)
    utils.weights_init(gen)

    optim_g=optim.Adam()


    loss=losses.GanLoss()
    loss_fm=losses.FeatureMatchingLoss()
    loss_p=losses.VGGLoss()

    dataloader = None

    for e in range(epoch):
        for ir, rgb, segment in tqdm(dataloader):

            condition=t.cat([rgb, segment], dim= ?)

            # Updating Generator

            ir_pred=gen(condition)

            out1_pred, out2_pred=disc(ir_pred)
            out1, out2=disc(ir)

            loss(out1_pred,out2_pred, is_real=False)
            loss(out1_pred,out2_pred, is_real=True)




            # Loss Calculation

            # Update (step)

        # epoch sonu bastırma ve kaydetme


if __name__ == '__main__':
    args = parser()
    opt = args.initialize()
    # TODO bunları main e ver kwargs?
    main()
