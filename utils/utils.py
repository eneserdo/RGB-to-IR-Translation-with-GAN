import torch as t
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import time, os, cv2
from skimage import io


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

        elif prefix == "ir" or prefix == "pred":
            mean = 0.35
            std = 0.18

        elif prefix == "segment":
            raise ValueError("Not implemented")

        else:
            raise TypeError("Name error")

        # image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu() * std + mean
        # image_unflat = image_tensor.detach().cpu()
        image_grid = make_grid(image_unflat, nrow=3)

        img = image_grid.permute(1, 2, 0).squeeze().numpy()
        img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8)

        img=cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)
        name = prefix + str(i) + r'.jpg'
        io.imsave(os.path.join(save_dir, name), img)


def save_model(gen, disc, e, save_dir):
    t.save(disc.state_dict(), os.path.join(save_dir, f"discriminator_{e}.pth"))
    t.save(gen.state_dict(), os.path.join(save_dir, f"generator_{e}.pth"))
    print("Models saved")


def show_loss(src_dir):
    pass