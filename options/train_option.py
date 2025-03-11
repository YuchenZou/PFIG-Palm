import argparse
import torchvision.transforms as transforms


def train_config():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders line, palm)')
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=25, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=1, help='number of CPU threads to use during batch generation')
    parser.add_argument('--lambda_pixel', type=float, default=5.0, help='weight for pixel loss')
    parser.add_argument('--lambda_feat', type=float, default=1.0, help='weight for feature loss')
    parser.add_argument('--lambda_KL', type=float, default=0.01, help='weight for KL divergence loss')
    parser.add_argument('--save_dir', type=str, default='output/train', help='directory to save train weight and images')
    parser.add_argument('--DINO_weight', type=str, default='models/DINO_weight.pth', help='directory to save train weight and images')
    return parser.parse_args()


def get_train_transforms(size):
    return [
        transforms.ToTensor(),
        transforms.Resize(int(size * 1.12)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]