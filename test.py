#!/usr/bin/env python3

import sys
import os
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.G_with_E import Generator
from data.datasets import ImageDataset
from options.test_option import test_config, get_test_transforms
from collections import OrderedDict

def main():
    # Parse the command-line options
    opt = test_config()
    print(opt)

    # Determine the device to use (GPU or CPU)
    device = torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Warn the user if CUDA is available but not enabled
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Load the pre-trained generator model
    netG = Generator(opt.input_nc, opt.output_nc).to(device)
    state_dict = torch.load(opt.G_weight, map_location=device)  # Load weights to the correct device
    
    # Remove 'module.' prefix if present (for DataParallel models)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # Remove 'module.' prefix
        new_state_dict[name] = v
    netG.load_state_dict(new_state_dict)
    netG.eval()

    # Create a DataLoader
    dataloader = DataLoader(
        ImageDataset(opt.dataroot, transforms_=get_test_transforms(opt.size), mode='test'),
        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu
    )

    # Define unnormalization for output images
    unnormalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )

    # Generate images with the generator model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            line = Variable(batch['A'].to(device))  # Move input to the correct device

            # Print progress to the console
            sys.stdout.write(f'\rGenerated images {i + 1:04d} of {len(dataloader):04d}')
            subfolder_name = f"{i:04d}"
            subfolder_path = os.path.join(opt.output_dir, subfolder_name)

            # Create a directory for saving generated images
            os.makedirs(subfolder_path, exist_ok=True)

            for num in range(opt.n_images):
                # Generate noise vector and create a fake image
                z = torch.randn(1, opt.z_dim, device=device)
                fake_palm, _, _ = netG(line, line, z)
                fake_palm = unnormalize(fake_palm)

                # Save the generated image
                img = transforms.ToPILImage()(fake_palm.squeeze(0).cpu())  # Convert to CPU before saving
                img.save(os.path.join(subfolder_path, f"{num:03d}.jpg"))

    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
