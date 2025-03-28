#!/usr/bin/env python3

import torch
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.G_with_E import Generator
from models.D_patch import Discriminator_Patch
from models.DINO import DinoNet
from util import LambdaLR, Logger, d_logistic_loss, g_nonsaturating_loss, set_requires_grad, save_generated_images
from data.datasets import ImageDataset
from models.IDI_module import IDI
from options.train_option import train_config, get_train_transforms
from torch.nn import DataParallel
from torch.nn.functional import cosine_similarity

def train(epoch, dataloader, netG, netD, netF, optimizer_G, optimizer_D, pix_loss, opt, logger):
    """Training loop for one epoch."""

    # Iterate over the dataset
    for i, batch in enumerate(dataloader):
        # Load input data and move to GPU if specified
        line = Variable(batch['A'].cuda() if opt.cuda else batch['A'])
        real_palm = Variable(batch['B'].cuda() if opt.cuda else batch['B'])

        # Train Discriminator
        set_requires_grad(netD, True)
        optimizer_D.zero_grad()

        # Generate fake images and get discriminator predictions
        fake_palm, mean, logvar = netG(line, real_palm)
        fake_pred = netD(fake_palm.detach())
        real_pred = netD(real_palm)

        # Calculate discriminator loss and update discriminator
        loss_D = d_logistic_loss(real_pred, fake_pred) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        set_requires_grad(netD, False)
        optimizer_G.zero_grad()

        # Calculate generator loss
        fake_pred = netD(fake_palm)
        loss_G = g_nonsaturating_loss(fake_pred)

        # Calculate pixel loss
        fake_fft_palm = IDI(line, real_palm)
        loss_pix = pix_loss(fake_palm, fake_fft_palm)

        # Calculate feature loss
        with torch.no_grad():
            line_feat = netF(line)
        palm_feat = netF(fake_palm)
        cos_similarity = cosine_similarity(line_feat, palm_feat).mean()
        loss_feat = (1. - cos_similarity) * 2.

        # Calculate KL loss
        loss_KL = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Combine losses and update generator
        loss_G_total = (loss_G
                        + loss_pix * opt.lambda_pixel
                        + loss_feat * opt.lambda_feat
                        + loss_KL * opt.lambda_KL)
        loss_G_total.backward()
        optimizer_G.step()

        # Logging and image saving  http://localhost:8097
        if i % 10 == 0:
            img_show = {'real_palm': real_palm, 'line': line, 'fake_palm': fake_palm}
        logger.log({
            'loss_G': loss_G, 'loss_D': loss_D, 'loss_feat': loss_feat,
            'loss_KL': loss_KL, 'loss_pix': loss_pix
        }, images=img_show)

        if i == 0:
            save_generated_images(opt.save_dir, epoch, line, fake_palm, real_palm)

    return loss_D, loss_G_total


def main():
    # Argument parsing
    opt = train_config()
    print(opt)

    # Check if CUDA is available
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Initialize models
    netG = Generator(opt.input_nc, opt.output_nc)
    netD = Discriminator_Patch(opt.output_nc)
    netF = DinoNet()
    netF.load_state_dict(torch.load(opt.DINO_weight)['net'])
    netF.eval()
    
    if opt.cuda:
        netG = DataParallel(netG).cuda()  # Wrap in DataParallel
        netD = DataParallel(netD).cuda()  # Wrap in DataParallel
        netF = DataParallel(netF).cuda()

    # Initialize loss function
    pix_loss = torch.nn.L1Loss()

    # Initialize optimizers
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Dataset loader
    dataloader = DataLoader(
        ImageDataset(opt.dataroot, transforms_=get_train_transforms(opt.size), mode='train'),
        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
    )

    # Logger setup
    logger = Logger(opt.n_epochs, len(dataloader))

    # Training loop
    for epoch in range(opt.epoch, opt.n_epochs):
        # Train for one epoch
        loss_D, loss_G_total = train(epoch, dataloader, netG, netD, netF, optimizer_G, optimizer_D,
                                     pix_loss, opt, logger)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        # Save models checkpoints
        if epoch>5 and epoch%2==0:
            save_model_checkpoint(opt.save_dir, netG, netD, epoch+1)


def save_model_checkpoint(checkpoint_dir, netG, netD, epoch):
    """Save model checkpoints."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(netG.state_dict(), os.path.join(checkpoint_dir, f'netG_epoch_{epoch}.pth'))
    torch.save(netD.state_dict(), os.path.join(checkpoint_dir, f'netD_epoch_{epoch}.pth'))


import os
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
