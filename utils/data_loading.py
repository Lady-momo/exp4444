import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
import os
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import pdb


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # 获取所有图像文件名（包含扩展名）
        self.image_files = [file for file in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, file)) and not file.startswith('.')]

        # 提取文件名主干（无扩展名）和原始扩展名
        self.ids = [splitext(f)[0] for f in self.image_files]
        self.extensions = [splitext(f)[1] for f in self.image_files]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        # pdb.set_trace()
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        original_ext = self.extensions[idx]  # 获取原始扩展名

        # 使用原始扩展名加载图像和掩码
        img_file = list(self.images_dir.glob(f"{name}{original_ext}"))
        mask_file = list(self.mask_dir.glob(f"{name}{self.mask_suffix}{original_ext}"))

        assert len(img_file) == 1, f'Expected 1 image file for {name}, found {len(img_file)}'
        assert len(mask_file) == 1, f'Expected 1 mask file for {name}, found {len(mask_file)}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img),
            'mask': torch.as_tensor(mask),
            'filename': name,
            'extension': original_ext  # 返回原始扩展名
        }




# # 图片大小设置
# img_size = (512, 512)

# # 数据预处理的转换
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# # 自定义数据集类
# class MyData(Dataset):
#     def __init__(self, path, img_size, transform):
#         self.path = path
#         self.files = os.listdir(self.path+"/images")
#         self.img_size = img_size
#         self.transform = transform

#     def __getitem__(self, index):
#         fn = self.files[index]
#         img = Image.open(self.path + "/images/" + fn).resize(self.img_size)
#         mask = Image.open(self.path + "/masks/" + fn).resize(self.img_size)
#         if self.transform:
#             img = self.transform(img)
#             mask = self.transform(mask)
#         return img, mask

#     def __len__(self):
#         return len(self.files)

# # 训练数据集
# train_path = "train"
# train_data = MyData(train_path, img_size, transform)
# train_data_size = len(train_data)
# train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

# # 测试数据集
# test_path = "test"
# test_data = MyData(test_path, img_size, transform)
# test_data_size = len(test_data)
# test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

# # 获取数据示例并可视化
# img, mask = train_data.__getitem__(0)
# img, mask = img.mul(255).byte(), mask.mul(255).byte()
# img, mask = img.numpy().transpose((1, 2, 0)), mask.numpy().transpose((1, 2, 0))

# # 可视化示例
# fig, ax = plt.subplots(1, 3, figsize=(30, 10))
# ax[0].imshow(img)
# ax[0].set_title('Img')
# ax[1].imshow(mask.squeeze())
# ax[1].set_title('Mask')
# ax[2].imshow(img)
# ax[2].contour(mask.squeeze(), colors='k', levels=[0.5])
# ax[2].set_title('Mixed')