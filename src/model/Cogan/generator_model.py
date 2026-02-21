import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("ConvTranspose") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class SharedEncoder(nn.Module):

    def __init__(self, in_ch=1, base_ch=64, n_down=2, n_res=4):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, base_ch, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        ch = base_ch
        for _ in range(n_down):
            layers += [
                nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(ch * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            ch *= 2

        for _ in range(n_res):
            layers += [ResBlock(ch)]

        self.model = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self, in_ch, out_ch=1, n_up=2, out_activation="sigmoid"):
        super().__init__()
        act = out_activation.lower().strip()
        assert act in ("sigmoid", "tanh", "none")

        layers = []
        ch = in_ch
        for _ in range(n_up):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(ch // 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            ch //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ch, out_ch, 7, 1, 0),
        ]

        self.model = nn.Sequential(*layers)
        self.act = nn.Sigmoid() if act == "sigmoid" else (nn.Tanh() if act == "tanh" else nn.Identity())

    def forward(self, x):
        return self.act(self.model(x))


class CoupledGenerator(nn.Module):

    def __init__(self, in_ch=1, base_ch=64, n_down=2, n_res=4, out_activation="sigmoid"):
        super().__init__()
        self.enc = SharedEncoder(in_ch, base_ch, n_down, n_res)
        self.dec_A = Decoder(self.enc.out_ch, out_ch=in_ch, n_up=n_down, out_activation=out_activation)
        self.dec_B = Decoder(self.enc.out_ch, out_ch=in_ch, n_up=n_down, out_activation=out_activation)

    def forward(self, x, to="B"):
        h = self.enc(x)
        if to.upper() == "A":
            return self.dec_A(h)
        else:
            return self.dec_B(h)
