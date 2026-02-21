import torch
import torch.nn as nn


class MultiDiscriminator(nn.Module):
    """
    BicycleGAN-style MultiDiscriminator:
    - 3 PatchGAN discriminators at different scales
    - input: 1-channel image (synthetic or real CT)
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()

        def disc_block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                f"disc_{i}",
                nn.Sequential(
                    *disc_block(in_channels, 64, normalize=False),
                    *disc_block(64, 128),
                    *disc_block(128, 256),
                    *disc_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1, bias=True),
                ),
            )
        # downsample
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        outputs = []
        for m in self.models:
            out = m(x)
            outputs.append(out)
            x = self.downsample(x)
        return outputs

    def compute_loss(self, x, is_real: bool):
        """
        is_real: True -> target=1, False -> target=0
        Loss LS-GAN multi-scala: sum MSE
        """
        targets_val = 1.0 if is_real else 0.0
        loss = 0.0
        for out in self.forward(x):
            target = torch.full_like(out, fill_value=targets_val, device=out.device)
            loss = loss + torch.mean((out - target) ** 2)
        return loss
