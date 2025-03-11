import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.mode = mode

        self.dir_A = os.path.join(root, "line")  # get the image directory
        if self.mode == 'train':
            self.dir_B = os.path.join(root, "palm")
        self.A_paths = sorted(make_dataset(self.dir_A))  # get image paths
        print(f"The line dataset contains {len(self.A_paths)} samples.")
        if self.mode == 'train':
            self.B_paths = sorted(make_dataset(self.dir_B))
            print(f"The palmprint dataset contains {len(self.B_paths)} samples.")

    def __getitem__(self, index):
        if self.mode == 'train':
            item_A = self.transform(Image.open(self.A_paths[index % len(self.A_paths)]))
            item_B = self.transform(Image.open(self.B_paths[random.randint(0, len(self.B_paths) - 1)]))
            return {'A': item_A, 'B': item_B}
        elif self.mode == 'test':
            item_A = self.transform(Image.open(self.A_paths[index % len(self.A_paths)]))
            return {'A': item_A}

    def __len__(self):
        return len(self.A_paths)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)