import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision.models import resnet18

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    """
    Downsampling block stile U-Net, with InstanceNorm
    """
    def __init__(self, in_size, out_size, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """
    Upsampling block stile U-Net
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), dim=1)
        return x


class Generator(nn.Module):
    """
    Generator conditional BicycleGAN:
    - input: image A (1 channel) + latent z (vector)
    - output: image B (1 channel)
    """
    def __init__(self, latent_dim: int, img_height: int, img_width: int,
                 in_channels: int = 1, out_channels: int = 1,
                 out_activation: str = "sigmoid"):
        super().__init__()
        self.h = img_height
        self.w = img_width
        self.in_channels = in_channels
        self.out_channels = out_channels

        act = out_activation.lower().strip()
        assert act in ("sigmoid", "tanh")
        self._out_act = act


        self.fc = nn.Linear(latent_dim, self.h * self.w)


        self.down1 = UNetDown(in_channels + 1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)

        # up

        self.up1 = UNetUp(512, 512)  # input d7 (512) → output 512, concat d6 (512) → 1024
        self.up2 = UNetUp(1024, 512)  # input u1 (1024) → output 512, concat d5 (512) → 1024
        self.up3 = UNetUp(1024, 512)  # input u2 (1024) → output 512, concat d4 (512) → 1024
        self.up4 = UNetUp(1024, 256)  # input u3 (1024) → output 256, concat d3 (256) → 512
        self.up5 = UNetUp(512, 128)  # input u4 (512) → output 128, concat d2 (128) → 256
        self.up6 = UNetUp(256, 64)  # input u5 (256) → output 64, concat d1 (64) → 128

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64 + 64, out_channels, 3, stride=1, padding=1, bias=True),
        )
        self._sigmoid = nn.Sigmoid()
        self._tanh = nn.Tanh()

    def forward(self, x, z):
        """
        x: (B,1,H,W)
        z: (B,latent_dim)
        """
        # z -> (B,1,H,W)
        z_map = self.fc(z).view(z.size(0), 1, self.h, self.w)

        d1 = self.down1(torch.cat((x, z_map), dim=1))  # (B,64, H/2, W/2)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        out = self.final(u6)
        if self._out_act == "sigmoid":
            return self._sigmoid(out)
        else:
            return self._tanh(out)


class Encoder(nn.Module):
    """
    VAE-style encoder: takes image B (1 channel) and produces mu, logvar
    """
    def __init__(self, latent_dim: int, in_channels: int = 1):
        super().__init__()


        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # adaptive pool -> vector 256
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):

        x = self.features(img)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class EncoderResNet18Mono(nn.Module):

    def __init__(self, latent_dim: int, pretrained: bool = False):
        super().__init__()


        backbone = resnet18(pretrained=pretrained)


        if pretrained:
            old_conv1 = backbone.conv1
            new_conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=(old_conv1.bias is not None),
            )
            with torch.no_grad():

                new_conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)
            backbone.conv1 = new_conv1
        else:

            backbone.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")


        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-3])


        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 256, 1, 1)

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img: torch.Tensor):

        x = self.feature_extractor(img)  # (B, 256, H', W')
        x = self.pool(x)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar