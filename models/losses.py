import torch.nn as nn
import torch as t
from .networks import Vgg19


# Perceptual Loss

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):

        x = t.cat([x, x, x], dim=1)
        y = t.cat([y, y, y], dim=1)

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# GAN Loss

class GanLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0):
        super(GanLoss, self).__init__()

        # One-sided smoothening
        self.smooth = target_real_label

        # Target tensors for first scale
        self.target1_tensor_ones = None
        self.target1_tensor_zeros = None

        # Target tensors for second scale
        self.target2_tensor_ones = None
        self.target2_tensor_zeros = None

        self.setted = False

        if use_lsgan:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def _set_tensors(self, output1, output2):

        self.target1_tensor_ones = t.ones_like(output1) * self.smooth
        self.target1_tensor_zeros = t.zeros_like(output1)

        self.target2_tensor_ones = t.ones_like(output2) * self.smooth
        self.target2_tensor_zeros = t.zeros_like(output2)

        print("Tensors Succesfully initializied")

        self.setted = True

    def forward(self, output1, output2, is_real=True):

        if not self.setted:
            self._set_tensors(output1, output2)

        if is_real:
            loss = self.criterion(self.target1_tensor_ones, output1) + self.criterion(self.target2_tensor_ones,
                                                                                      output2)
        else:
            loss = self.criterion(self.target1_tensor_zeros, output1) + self.criterion(self.target2_tensor_zeros,
                                                                                       output2)

        return loss


# Feature Matching Loss

class FeatureMatchingLoss():
    def __init__(self):
        # Feature Matching Loss
        self.L1 = nn.L1Loss()
        self.w = [1./8, 1./4, 1./2, 1.]

    def __call__(self, output1, output2):
        loss = 0
        for i in range(len(output1)):
            loss += self.L1(output1[i], output2[i].detach()) * self.w[i]

        return loss
