import torch as t
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import time, cv2, os


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def show_tensor_images(image_tensor, i, save_dir, prefix, resize_factor=0.5):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat, nrow=3)

    img = image_grid.permute(1, 2, 0).squeeze().numpy()
    img = np.clip(img * 255, a_min=0, a_max=255).astype('int')

    img=cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)
    name = prefix + str(i) + r'.jpg'
    cv2.imwrite(os.path.join(save_dir, name), img)

