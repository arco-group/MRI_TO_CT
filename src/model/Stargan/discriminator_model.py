import torch.nn as nn
import torch


class Discriminator(nn.Module):

    def __init__(self, img_shape=(1, 256, 256), c_dim: int = 2, n_strided: int = 6):
        super().__init__()
        channels, img_h, img_w = img_shape
        assert channels == 1, "Discriminator StarGAN_ale è pensato per 1 canale."

        def disc_block(in_filters, out_filters):
            return [
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                nn.LeakyReLU(0.01, inplace=True),
            ]

        layers = disc_block(channels, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(disc_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out_adv = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # Output 2: class prediction (c_dim)
        kernel_size = img_h // (2 ** n_strided)
        self.out_cls = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

    def forward(self, img):
        feat = self.model(img)
        out_adv = self.out_adv(feat)                # [B,1,h',w']
        out_cls = self.out_cls(feat)                # [B,c_dim,1,1] (se kernel_size combacia)
        out_cls = out_cls.view(out_cls.size(0), -1) # [B,c_dim]
        return out_adv, out_cls
