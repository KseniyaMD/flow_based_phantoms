from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import random
import os
import cv2 as cv
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd



def get_imgs_from_dir(path_a, path_b):
    list_a = []
    for root, dir, files in os.walk(path_a):
        for file in files:
            list_a.append(f'{path_a}/{file}')

    list_b = []
    list_b.append(path_b)
    list_b = list_b * len(list_a)

    return list_a, list_b

class PadToSize:
    def __init__(self, size, fill=0):
        if isinstance(size, int):
            self.target_h = self.target_w = size
        else:
            self.target_h, self.target_w = size
        self.fill = fill

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3:
            raise ValueError(f"Error {img.shape}")
        c, h, w = img.shape

        pad_h = max(self.target_h - h, 0)
        pad_w = max(self.target_w - w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=self.fill)
        return padded


def normalize_per_image(img: np.ndarray, eps=1e-8):

    img_c = img.copy()
    img_c = img_c.astype(np.float32)

    norm_img = np.zeros_like(img_c)

    H, W, C = img.shape
    min_vals = np.zeros((C,), dtype=np.float32)
    max_vals = np.zeros((C,), dtype=np.float32)

    # Для каждого канала отдельно
    for c in range(C):
        channel = img_c[:, :, c].copy()
        cmin = channel.min()
        cmax = channel.max()
        min_vals[c] = cmin
        max_vals[c] = cmax
        denom = (cmax - cmin + eps)


        norm_img[:, :, c] = (channel - cmin) / denom


    return norm_img, min_vals, max_vals


class BaseDataset(data.Dataset):

    def __init__(self, cfg):

        rootA = cfg.rootA
        rootB = cfg.rootB
        self.return_name = cfg.return_name

        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Resize((250, 250)))
        #transform_list.append(PadToSize((256, 512)))
        transform_list.append(PadToSize((256, 256)))
        transform = transforms.Compose(transform_list)

        transform_list_2 = []
        transform_list_2.append(transforms.ToTensor())
        transform_2 = transforms.Compose(transform_list_2)

        transform_list_3 = []
        transform_list_3.append(transforms.ToTensor())
        transform_list_3.append(transforms.Resize((128, 128)))
        transform_3 = transforms.Compose(transform_list_3)


        imgsA = []
        imgsB = []


        imgs_A_1, imgs_B_1 = get_imgs_from_dir(rootA, rootB)

        imgsA += imgs_A_1
        imgsB += imgs_B_1

        random.shuffle(imgs_A_1)
        random.shuffle(imgs_B_1)

        total_len = len(imgsA)
        self.rootA = rootA
        self.rootB = rootB
        self.imgsA = imgsA
        self.imgsB = imgsB

        self.transform = transform
        self.transform_2 = transform_2
        self.transform_3 = transform_3

    def __getitem__(self, index):
        # index = 0
        pathA = self.imgsA[index]
        pathB = self.imgsB[index]
        imgA = np.load(pathA)
        imgA_not_normalized = imgA.copy()
        imgB = np.load(pathB)
        imgA_not_resized = imgA.copy()
        imgA_only_resized = imgA.copy()

        pd = imgB[:, :, 0]
        t1 = imgB[:, :, 1] * 1e-3
        t2 = imgB[:, :, 2] * 1e-3

        maps_list = [t1, t2, pd]
        maps_arr = np.array(maps_list)
        imgB = maps_arr.transpose(1, 2, 0)

        imgA, imgA_min_vals, imgA_max_vals = normalize_per_image(imgA, eps=1e-8)
        imgB, _, _ = normalize_per_image(imgB, eps=1e-8)

        imgA_not_resized, _, _ = normalize_per_image(imgA_not_resized, eps=1e-8)

        name = pathA.split('/')[-1].split('_')[0]

        imgA = self.transform(imgA)
        imgB = self.transform(imgB)
        imgA_not_normalized = self.transform_2(imgA_not_normalized)
        imgA_normalized_not_resized = self.transform_2(imgA_not_resized)
        imgA_normalized_resized_128 = self.transform_3(imgA_not_resized)

        imgA_only_resized = self.transform_3(imgA_only_resized)

        if not self.return_name:
            return imgA, imgB, name
        else:
            return imgA, imgB, imgA_not_normalized, imgA_max_vals, imgA_only_resized, name

    def __len__(self):
        return len(self.imgsA)


def get_dataset(cfg):
    dataset = BaseDataset(cfg)
    return dataset