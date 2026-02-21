import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
        if getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):

    def __init__(self, img_channels=1, latent_dim=8, n_residual_blocks=6, out_activation="sigmoid"):
        super().__init__()
        assert out_activation in ("sigmoid", "tanh")
        self.out_activation = out_activation
        self.latent_dim = latent_dim
        self.img_channels = img_channels

        # proietta z in immagine
        self.fc = nn.Linear(latent_dim, img_channels * 256 * 256)

        self.l1 = nn.Sequential(
            nn.Conv2d(img_channels * 2, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )

        resblocks = [ResidualBlock(64) for _ in range(n_residual_blocks)]
        self.resblocks = nn.Sequential(*resblocks)

        self.l2 = nn.Conv2d(64, img_channels, 3, 1, 1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, z):
        # z -> (B, C, H, W)
        z_img = self.fc(z).view(img.size(0), self.img_channels, img.size(2), img.size(3))
        gen_input = torch.cat((img, z_img), dim=1)

        out = self.l1(gen_input)
        out = self.resblocks(out)
        out = self.l2(out)

        if self.out_activation == "tanh":
            return self.tanh(out)
        return self.sigmoid(out)
