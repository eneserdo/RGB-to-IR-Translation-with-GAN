import torch.nn as nn
import torch as t
from collections import namedtuple


# First

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])

# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)

# Second

class Vgg16(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg16, self).__init__()
        self.net = models.vgg16(pretrained).features.eval()

    def forward(self, x):
        out = []
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i in [3, 8, 15, 22, 29]:
                # print(self.net[i])
                out.append(x)
        return out


class GanLoss(nn.Module):
    def __init__(self, use_lsgan, target_real_label=1.0):
        super(GanLoss, self).__init__()

        # One-sided smoothening
        self.smooth = target_real_label

        # Target tensors for first scale
        self.target1_tensor_ones = None
        self.target1_tensor_zeros = None

        # Target tensors for second scale
        self.target2_tensor_ones = None
        self.target2_tensor_zeros = None

        if use_lsgan:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def set_tensors(self, output1, output2):

        self.target1_tensor_ones = t.ones_like(output1[-1]) * self.smooth
        self.target1_tensor_zeros = t.zeros_like(output2[-1])

        self.target2_tensor_ones = t.ones_like(output2[-1]) * self.smooth
        self.target2_tensor_zeros = t.zeros_like(output2[-1])

        print("Tensors Succesfully initializied")
        return None

    def calc_GAN(self, output1, output2, is_real=True):

        loss = 0
        if is_real:
            loss = self.criterion(self.target1_tensor_ones, output1[-1]) + self.criterion(self.target2_tensor_ones,
                                                                                          output2[-1])
        else:
            loss = self.criterion(self.target1_tensor_zeros, output1[-1]) + self.criterion(self.target2_tensor_zeros,
                                                                                           output2[-1])

        return loss


class FeatureMatchingLoss():
    def __init__(self):

        # Feature Matching Loss
        self.L1 = nn.L1Loss()
        self.w = [1 / 32., 1 / 16., 1 / 8., 1 / 4., 1 / 2.]

    def calc_FM(self, output1, output2):
        loss = 0
        for i in range(1, len(output1)):
            loss += self.L1(output1[i], output2[i]) * self.w[i]

        return loss

