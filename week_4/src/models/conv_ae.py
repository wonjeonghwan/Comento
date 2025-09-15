import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128,4, 2, 1), nn.ReLU(True),
            nn.Conv2d(128,256,4, 2, 1), nn.ReLU(True),
        )
        self.enc_fc = nn.Sequential(nn.Flatten(), nn.Linear(256*8*8, latent_dim))
        self.dec_fc = nn.Sequential(nn.Linear(latent_dim, 256*8*8), nn.Unflatten(1, (256,8,8)))
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(128,64, 4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3,  4,2,1), nn.Tanh()
        )

    def encode(self, x):
        return self.enc_fc(self.enc(x))

    def decode(self, z):
        return self.dec(self.dec_fc(z))

    def forward(self, x):
        return self.decode(self.encode(x))
