import cv2
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from skimage import io
from torchvision.utils import make_grid


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def save_tensor_images(image_tensor, i, save_dir, prefix, resize_factor=0.5):
    with t.no_grad():
        if prefix == "rgb":
            mean = t.tensor([0.34, 0.33, 0.35]).reshape(1, 3, 1, 1)
            std = t.tensor([0.19, 0.18, 0.18]).reshape(1, 3, 1, 1)

        elif prefix == "ir":
            mean = 0.35
            std = 0.18

        elif prefix == "pred":
            mean = 0.5
            std = 0.5


        elif prefix == "segment":
            raise ValueError("Not implemented")

        else:
            raise TypeError("Name error")


        print(max(image_tensor.detach().cpu()))
        print(min(image_tensor.detach().cpu()))

        # image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu() * std + mean
        # image_unflat = image_tensor.detach().cpu()

        image_grid = make_grid(image_unflat, nrow=3)

        img = image_grid.permute(1, 2, 0).squeeze().numpy()
        img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8)

        img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        name = prefix + str(i) + r'.jpg'
        io.imsave(os.path.join(save_dir, name), img)

def save_all_images(rgb, ir, pred, i, save_dir, resize_factor=0.5):
    with t.no_grad():

        mean_rgb = t.tensor([0.34, 0.33, 0.35]).reshape(1, 3, 1, 1)
        std_rgb = t.tensor([0.19, 0.18, 0.18]).reshape(1, 3, 1, 1)

        mean_ir = 0.35
        std_ir = 0.18

        ir_n = ir.detach().cpu() * std_ir + mean_ir

        pred_n = pred.detach().cpu() * std_ir + mean_ir

        rgb_n = rgb.detach().cpu() * std_rgb + mean_rgb

        image_unflat = t.cat([pred_n, ir_n, rgb_n], dim=0)

        image_grid = make_grid(image_unflat, nrow=2)

        img = image_grid.permute(1, 2, 0).squeeze().numpy()
        img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8)

        img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

        io.imsave(os.path.join(save_dir, f"super{i:0>4d}.jpg"), img)



def save_model(disc, gen, cur_epoch, save_dir):
    t.save(disc.state_dict(), os.path.join(save_dir, f"e_{cur_epoch:0>3d}_discriminator.pth"))
    t.save(gen.state_dict(), os.path.join(save_dir, f"e_{cur_epoch:0>3d}_generator.pth"))
    print("Models saved")


def show_loss(src_dir):
    files_d = glob.glob(os.path.join(src_dir, 'v*_d_loss.npy'))
    sorted_disc_loss = sorted(files_d)

    files_g = glob.glob(os.path.join(src_dir, 'v*_g_loss.npy'))
    sorted_gen_loss = sorted(files_g)

    print('Reading all saved losses...')
    # print(sorted_disc_loss)
    arr_g = np.array([])
    arr_d = np.array([])

    for l in range(len(sorted_gen_loss)):
        arr_g = np.concatenate([arr_g, np.load(sorted_gen_loss[l])])

        arr_d = np.concatenate([arr_d, np.load(sorted_disc_loss[l])])

    plt.plot(np.arange(arr_d.shape[0]), arr_d, color='orange', label='Discriminator')
    plt.plot(np.arange(arr_g.shape[0]), arr_g, color='blue', label='Generator')
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()


def save_loss(d, d1, d2, g, g1, g2, fm1, fm2, p, path, e):
    np.save(os.path.join(path, f'v{e:0>3d}_d_loss.npy'), np.array(d))
    np.save(os.path.join(path, f'v{e:0>3d}_d1_loss.npy'), np.array(d1))
    np.save(os.path.join(path, f'v{e:0>3d}_d2_loss.npy'), np.array(d2))

    np.save(os.path.join(path, f'v{e:0>3d}_g_loss.npy'), np.array(g))
    np.save(os.path.join(path, f'v{e:0>3d}_g1_loss.npy'), np.array(g1))
    np.save(os.path.join(path, f'v{e:0>3d}_g2_loss.npy'), np.array(g2))

    np.save(os.path.join(path, f'v{e:0>3d}_fm1_loss.npy'), np.array(fm1))
    np.save(os.path.join(path, f'v{e:0>3d}_fm2_loss.npy'), np.array(fm2))
    np.save(os.path.join(path, f'v{e:0>3d}_p_loss.npy'), np.array(p))


def lr_lambda(epoch, decay_after=100):
    ''' Function for scheduling learning '''
    return 1. if epoch < decay_after else 1 - float(epoch - decay_after) / (200 - decay_after)


