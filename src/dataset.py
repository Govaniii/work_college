import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    def calculate_mean(self, images):
        mean_image = np.mean(images, axis=0)
        return mean_image

    def __init__(self, root_dir, type, rgb_transform=None, depth_transform=None):
        self.root_dir = root_dir
        self.type = type
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        self.rgb_folder = os.path.join(root_dir, 'rgb')
        self.depth_folder = os.path.join(root_dir, 'depth')

        # Assuming the number of images is the same for rgb and depth
        self.num_images = len(os.listdir(self.rgb_folder))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        rgb_file = f'_rgb_{idx}_.jpg'
        depth_file = f'_depth_{idx}.jpg'

        rgb_path = os.path.join(self.rgb_folder, rgb_file)
        depth_path = os.path.join(self.depth_folder, depth_file)

        rgb_image = Image.open(rgb_path)
        depth_image = Image.open(depth_path)

        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        if self.depth_transform:
            depth_image = self.depth_transform(depth_image)

        sample = {'rgb': rgb_image, 'depth': depth_image}
        return sample

# Пример использования:
rgb_data_transforms = transforms.Compose([transforms.ToTensor()])
depth_data_transforms = transforms.Compose([transforms.ToTensor()])

# Путь к вашим данным
nyu_data_path = 'dataset'

train_loader = DataLoader(
    CustomDataset(nyu_data_path, 'training', rgb_transform=rgb_data_transforms, depth_transform=depth_data_transforms),
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    CustomDataset(nyu_data_path, 'validation', rgb_transform=rgb_data_transforms, depth_transform=depth_data_transforms),
    batch_size=32,
    shuffle=False,
    num_workers=0
)
