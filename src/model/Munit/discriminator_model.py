import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDiscriminator(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                f"disc_{i}",
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1),
                ),
            )


        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, count_include_pad=False)

    def compute_loss(self, x, gt_scalar: float):

        loss = 0.0
        for out in self.forward(x):
            target = torch.full_like(out, fill_value=gt_scalar)
            loss = loss + F.mse_loss(out, target)
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs
