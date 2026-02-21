import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class SharedDiscriminatorTrunk(nn.Module):
    """
    Trunk condiviso stile CoGAN (PatchGAN).
    """
    def __init__(self, in_ch=1, base_ch=64, n_layers=4):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        ch = base_ch
        for _ in range(1, n_layers):
            layers += [
                nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(ch * 2, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch *= 2
        self.model = nn.Sequential(*layers)
        self.out_ch = ch


class CoupledDiscriminator(nn.Module):

    def __init__(self, in_ch=1, base_ch=64, n_layers=4):
        super().__init__()
        self.trunk = SharedDiscriminatorTrunk(in_ch, base_ch, n_layers)
        self.head_A = nn.Conv2d(self.trunk.out_ch, 1, 3, 1, 1, bias=False)
        self.head_B = nn.Conv2d(self.trunk.out_ch, 1, 3, 1, 1, bias=False)

    def forward(self, x, domain="B"):
        feat = self.trunk.model(x)
        if domain.upper() == "A":
            return self.head_A(feat)
        else:
            return self.head_B(feat)
