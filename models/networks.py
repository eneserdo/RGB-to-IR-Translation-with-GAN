import torch as t
import torch.nn as nn
from torchvision import models


####### Generator #######

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=7, norm_layer=nn.InstanceNorm2d, padding_type='reflect', transposed=False):

        assert (n_blocks >= 0)
        super(Generator, self).__init__()
        # activation = nn.ReLU(True)
        activation = nn.LeakyReLU()

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if transposed:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                            output_padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
            else:
                model += [nn.Upsample(scale_factor=2),nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.LeakyReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


####### Discriminator #######
class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, norm=nn.BatchNorm2d):
        super(Discriminator, self).__init__()

        # TODO: norm is not implemented

        self.d1 = self._conv_block(input_nc, ndf, k=5, norm=norm, p=2)

        self.d2 = self._conv_block(ndf, ndf, k=3, norm=norm, pool=True)

        self.d3 = self._conv_block(ndf, ndf * 2, k=3, norm=norm)

        self.d4 = self._conv_block(ndf * 2, ndf * 4, k=3, norm=norm, pool=True, drop=True)

        self.d5 = self._conv_block(ndf * 4, ndf * 4, k=3, norm=norm)

        self.d6 = self._conv_block(ndf * 4, 1, k=3, norm=norm, pool=True, drop=True)

    def _conv_block(self, in_ch, out_ch, k, s=1, p=1, norm=nn.BatchNorm2d, pool=False, drop=False):

        layers = [nn.Conv2d(in_ch, out_ch, k, padding=p, stride=s),
                  nn.BatchNorm2d(out_ch),
                  nn.LeakyReLU()]

        if pool:
            layers.append(nn.MaxPool2d(2, stride=2))

        if drop:
            layers.append(nn.Dropout2d(0.2))

        return nn.Sequential(*layers)

    def forward(self, x):
        layers = [self.d1, self.d2, self.d3, self.d4, self.d5, self.d6]

        fm = [x]

        for layer in layers:
            fm.append(layer(fm[-1]))

        return fm[2:]


class MultiScaleDisc(nn.Module):
    def __init__(self, input_nc=2, ndf=64, norm=nn.BatchNorm2d):
        super(MultiScaleDisc, self).__init__()

        self.disc1=Discriminator(input_nc, ndf, norm)
        self.disc2=Discriminator(input_nc, ndf, norm)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        fm1=self.disc1(x)
        fm2=self.disc2(self.downsample(x))

        return fm1, fm2


####### VGG for perceptual loss #######

class Vgg19(t.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = t.nn.Sequential()
        self.slice2 = t.nn.Sequential()
        self.slice3 = t.nn.Sequential()
        self.slice4 = t.nn.Sequential()
        self.slice5 = t.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Alternative implementation from pix2pix

class Discriminator_v0(nn.Module):
    def __init__(self, in_channel=1, ndf=64):

        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(self.conv_block(in_channel, ndf),
                                  self.conv_block(ndf, 2*ndf, dropout=True),
                                  self.conv_block(2*ndf, 4*ndf),
                                  self.conv_block(4*ndf, 8*ndf),
                                  nn.Conv2d(8*ndf, 1, 4, padding=1))


    def conv_block(self, in_ch: int, out_ch: int, dropout: bool = False, bn: bool = True):
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                  nn.BatchNorm2d(out_ch),
                  nn.LeakyReLU(0.2)]

        if not bn:
            layers.pop(1)

        if dropout:
            layers.insert(-1, nn.Dropout(0.5))

        return nn.Sequential(*layers)

    def forward(self, image):
        return self.disc(image)




# U-NET

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(Encoder, self).__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(out_ch),
                  nn.LeakyReLU(0.2)]
        if not bn:
            layers.pop(1)

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super(Decoder, self).__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(out_ch),
                  nn.ReLU()]

        if dropout:
            layers.insert(-1, nn.Dropout(0.5))

        self.decode = nn.Sequential(*layers)

    def forward(self, x1, x2):
        if x2 is not None:
            return self.decode(t.cat([x1, x2], dim=1))

        else:
            return self.decode(x1)


class Generator_v0(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        # type: (int, int) -> None

        super(Generator, self).__init__()
        self.encoder1 = Encoder(in_channel, 64, bn=False)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.encoder4 = Encoder(256, 512)
        self.encoder5 = Encoder(512, 512)
        self.encoder6 = Encoder(512, 512)
        self.encoder7 = Encoder(512, 512)
        self.encoder8 = Encoder(512, 512)

        self.decoder9 = Decoder(512, 512, dropout=True)
        self.decoder10 = Decoder(1024, 512, dropout=True)
        self.decoder11 = Decoder(1024, 512, dropout=True)
        self.decoder12 = Decoder(1024, 512)
        self.decoder13 = Decoder(1024, 256)
        self.decoder14 = Decoder(512, 128)
        self.decoder15 = Decoder(256, 64)
        self.last = nn.Sequential(nn.Conv2d(128, out_channel, 1),
                                  nn.Tanh())

    def forward(self, image):

        e1 = self.encoder1(image)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)

        x = self.decoder9(e8, None)
        x = self.decoder10(x, e7)
        x = self.decoder11(x, e6)
        x = self.decoder12(x, e5)
        x = self.decoder13(x, e4)
        x = self.decoder14(x, e3)
        x = self.decoder15(x, e2)

        x = self.last(t.cat([x, e1], dim=1))

        return x
