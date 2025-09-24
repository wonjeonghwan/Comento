import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # (3, 64, 64) -> (16, 32, 32)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # (64, 8, 8)
            nn.ReLU(),
        )
        # 디코더
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()  # 픽셀값 0~1
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
