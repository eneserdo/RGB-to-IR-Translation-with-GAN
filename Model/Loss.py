import torch.nn as nn
import torch.utils.model_zoo as models
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

