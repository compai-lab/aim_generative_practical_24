import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms, datasets
import torchvision.utils as vutils
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(128,128)):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(128,128)):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN(pl.LightningModule):
    def __init__(self, latent_dim, img_shape, lr=0.0002, b1=0.5, b2=0.999):
        super(GAN, self).__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)
        self.automatic_optimization = False

        self.validation_z = torch.randn(8, latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return nn.BCELoss()(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs = batch

        optimizer_g, optimizer_d = self.optimizers()

        valid = torch.ones(imgs.size(0), 1).type_as(imgs)
        fake = torch.zeros(imgs.size(0), 1).type_as(imgs)

        # Train Generator
        z = torch.randn(imgs.size(0), self.hparams.latent_dim).type_as(imgs)
        generated_imgs = self(z)
        g_loss = self.adversarial_loss(self.discriminator(generated_imgs), valid)

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()

        # Train Discriminator
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()

        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d]

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self(z)
        grid = vutils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
