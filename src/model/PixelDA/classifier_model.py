import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, in_channels=1, base_dim=64, n_classes=3, img_dim=256):
        super().__init__()

        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.features = nn.Sequential(
            *block(in_channels, base_dim, norm=False),
            *block(base_dim, base_dim * 2),
            *block(base_dim * 2, base_dim * 4),
            *block(base_dim * 4, base_dim * 8),
        )


        ds = img_dim // (2 ** 4)
        self.head = nn.Linear(base_dim * 8 * ds * ds, n_classes)

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        return self.head(f)
