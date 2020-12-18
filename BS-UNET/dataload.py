from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np


def make_dataset(root):
    imgs = []
    n = len(os.listdir(os.path.join(root, "train_data")))
    for i in range(n):
        img = os.path.join(root, "train_data", "%d.npy" % i)
        mask = os.path.join(root, "train_label", "%d.npy" % i)
        imgs.append((img, mask))
    return imgs


class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.imgs = make_dataset(root)[0:1000]
        if train == 0:
            self.imgs = make_dataset(root)[1000:1213]
        self.imgs=make_dataset(root)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = np.load(x_path)
        img_y = np.load(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':

    make_dataset('data')

