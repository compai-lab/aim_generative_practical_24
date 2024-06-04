import torch
import torch.nn as nn
import torch.autograd as autograd
import pytorch_lightning as pl

class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.init_size = opt['img_size'] // 4
        self.l1 = nn.Sequential(nn.Linear(opt['latent_dim'], 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt['channels'], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt['channels'], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        ds_size = opt['img_size'] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.adv_layer(features)
        return validity

    def forward_features(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        return features

class WGAN(nn.Module):
    def __init__(self, image_size=128, latent_dim=100, channels=1):
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        opt = {'img_size': image_size, 'latent_dim': latent_dim, 'channels': channels}
        self.generator = Generator(opt)
        self.discriminator = Discriminator(opt)

    def forward(self, x):
        return self.generator(x)

class GANModule(pl.LightningModule):
    def __init__(self, hparams):
        super(GANModule, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = WGAN(hparams['image_size'], hparams['latent_dim'], hparams['channels'])
        self.automatic_optimization = False
        self.lambda_gp = 10

    def forward(self, z):
        return self.model.generator(z)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.model.discriminator(interpolates)
        fake = torch.ones(d_interpolates.size(), device=self.device)
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=fake, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx):
        imgs = batch
        batch_size = imgs.size(0)
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        g_opt, d_opt = self.optimizers()

        # Train Discriminator
        d_opt.zero_grad()
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        fake_imgs = self.model.generator(z)
        real_validity = self.model.discriminator(imgs)
        fake_validity = self.model.discriminator(fake_imgs)
        gradient_penalty = self.compute_gradient_penalty(imgs, fake_imgs)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
        self.manual_backward(d_loss)
        d_opt.step()

        # Train Generator
        if batch_idx % self.hparams.n_critic == 0:
            g_opt.zero_grad()
            fake_imgs = self.model.generator(z)
            fake_validity = self.model.discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)
            self.manual_backward(g_loss)
            g_opt.step()

        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]
