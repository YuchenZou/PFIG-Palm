import argparse
import torchvision.transforms as transforms


def test_config():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description="Testing Script")
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=1, help='number of CPU threads to use during batch generation')
    parser.add_argument('--G_weight', type=str, default='output/netG_best.pth', help='A2B generator checkpoint file')
    parser.add_argument('--n_images', type=int, default=40, help='number of images to generate per input')
    parser.add_argument('--z_dim', type=int, default=8, help='dimension of the noise vector z')
    parser.add_argument('--output_dir', type=str, default='output/val', help='directory to save generated images')
    return parser.parse_args()

def get_test_transforms(size):
    return [
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
