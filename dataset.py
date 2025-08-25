import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt


class PairedImgDataset(Dataset):
    def __init__(self, root, tag='train', img_size=128):
        super().__init__()
        self.img_size = img_size
        self.upscale_factor = 1
        self.dir = os.path.join(root)
        self.vi_files = sorted(os.listdir(os.path.join(self.dir, 'MRI')))  #Gray
        self.ir_files = sorted(os.listdir(os.path.join(self.dir, 'CT')))  #color RGB

        assert len(self.vi_files) == len(self.ir_files), "vi numbers doesn't match ir number."
        self.data_size = len(self.vi_files)
        self.tag = tag

    def __getitem__(self, index):
        vi_nm, ir_nm = self.vi_files[index], self.ir_files[index]
        vi_img = Image.open(os.path.join(self.dir, 'MRI', vi_nm)).convert('L')
        ir_img = Image.open(os.path.join(self.dir, 'CT', ir_nm)).convert('L')

        vi_img = TF.to_tensor(vi_img)
        ir_img = TF.to_tensor(ir_img)

        img_h = vi_img.shape[1]
        img_w = vi_img.shape[2]
        if img_h - self.img_size == 0:
            r, c = 0, 0
        else:
            r = np.random.randint(0, img_h - self.img_size)
            c = np.random.randint(0, img_w - self.img_size)
        vi_img = vi_img[:, r: r + self.img_size, c: c + self.img_size]
        ir_img = ir_img[:, r: r + self.img_size, c: c + self.img_size]

        return ir_img, vi_img

    def __len__(self):
        return self.data_size

def get_train_loader(data_dir1, batch_size, num_workers, shuffle=True, img_size=128):
    train_dataset = PairedImgDataset(data_dir1, tag='train', img_size=img_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=True,
                              num_workers=num_workers,
                              drop_last=True)

    return train_loader, len(train_dataset)


def main():

    data_dir = "Medical_dataset_test/MRI-CT"  

    batch_size = 8
    num_workers = 8  
    img_size = 256 

    train_loader, dataset_size = get_train_loader(data_dir, batch_size, num_workers, shuffle=True, img_size=img_size)

    print(f"Total samples in the dataset: {dataset_size}")

    for batch_idx, (ir_img, vi_img) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"ir image shape: {ir_img.shape}")
        print(f"vi image shape: {vi_img.shape}")

        if batch_idx >= 10:
            break

if __name__ == "__main__":
    main()