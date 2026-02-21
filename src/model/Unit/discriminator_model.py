import torch
import torch.nn as nn


class _DiscBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: bool = True, stride: int = 2):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            _DiscBlock(in_channels, 64, norm=False, stride=2),  # 128x128
            _DiscBlock(64, 128, norm=True, stride=2),           # 64x64
            _DiscBlock(128, 256, norm=True, stride=2),          # 32x32
            _DiscBlock(256, 512, norm=True, stride=1),          # 32x32 -> patch più denso
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")  # logits
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    x = torch.randn(2, 1, 256, 256)
    d = Discriminator(1)
    y = d(x)
    print("D out:", y.shape)  # ~ (2,1,32,32)
