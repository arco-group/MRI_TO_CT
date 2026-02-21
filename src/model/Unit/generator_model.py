import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block con InstanceNorm e ReflectionPad."""
    def __init__(self, features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3, bias=False),
            nn.InstanceNorm2d(features),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):

    def __init__(self, in_channels: int = 1, dim: int = 64, n_downsample: int = 2, shared_block: nn.Module = None):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7, bias=False),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2


        for _ in range(3):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.Sequential(*layers)
        self.shared_block = shared_block  # ResidualBlock(features=shared_dim)

    @staticmethod
    def _reparameterization(mu: torch.Tensor) -> torch.Tensor:
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        eps = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return mu + eps

    def forward(self, x):
        x = self.model_blocks(x)
        mu = self.shared_block(x) if self.shared_block is not None else x
        z = self._reparameterization(mu)
        return mu, z



class Generator(nn.Module):

    def __init__(
        self,
        out_channels: int = 1,
        dim: int = 64,
        n_upsample: int = 2,
        shared_block: nn.Module = None,
        out_activation: str = 'sigmoid',
    ):
        super().__init__()
        self.shared_block = shared_block
        act = out_activation.lower().strip()
        assert act in ('sigmoid', 'tanh'), "out_activation would be 'sigmoid' o 'tanh'"
        self._out_act = act


        ch = dim * (2 ** n_upsample)

        layers = []

        for _ in range(3):
            layers += [ResidualBlock(ch)]

        # upsample
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ch // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch //= 2


        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ch, out_channels, 7, bias=True),
        ]

        self.model_blocks = nn.Sequential(*layers)
        self._final_sigmoid = nn.Sigmoid()
        self._final_tanh = nn.Tanh()

    def forward(self, x):
        x = self.shared_block(x) if self.shared_block is not None else x
        x = self.model_blocks(x)
        if self._out_act == 'sigmoid':
            return self._final_sigmoid(x)
        else:
            return self._final_tanh(x)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    shared_dim = 64 * (2 ** 2)
    shared_E = ResidualBlock(features=shared_dim).to(device)
    shared_G = ResidualBlock(features=shared_dim).to(device)

    E = Encoder(in_channels=1, dim=64, n_downsample=2, shared_block=shared_E).to(device)
    G = Generator(out_channels=1, dim=64, n_upsample=2, shared_block=shared_G, out_activation='sigmoid').to(device)

    x = torch.randn(2, 1, 256, 256).to(device)
    mu, z = E(x)
    y = G(z)

    print("mu:", mu.shape, "z:", z.shape, "y:", y.shape)  # atteso: (2, 256, 64, 64) -> up → (2,1,256,256)
