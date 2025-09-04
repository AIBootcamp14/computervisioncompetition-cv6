import os

import cv2
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


import os, cv2, torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def recipe_base(img_size=384):
    """기존 학습 파이프라인과 성격 동일(약한 기하 + 조도 + 압축)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Affine(rotate=(-8, 8), shear={'x':(-4, 4), 'y':(-4, 4)}, p=0.5),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
        A.ImageCompression(quality_lower=70, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def recipe_lowq(img_size=384):
    """저화질/노이즈 환경을 모사(블러/다운스케일/압축)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Affine(rotate=(-6, 6), shear={'x':(-3, 3), 'y':(-3, 3)}, p=0.5),
        A.Perspective(scale=(0.02, 0.04), p=0.25),
        A.ImageCompression(quality_lower=55, quality_upper=90, p=0.7),
        A.Downscale(scale_min=0.90, scale_max=0.98, p=0.3),
        A.GaussianBlur(blur_limit=(3,5), p=0.3),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.25),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_train_recipes(img_size=384):
    """학습에 사용할 레시피 목록(2개) 반환 → 데이터셋 길이 2배"""
    return [recipe_base(img_size), recipe_lowq(img_size)]


def get_val_transform(img_size=384):
    """검증/추론용(증강 없음)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

class MultiRecipeDataset(Dataset):
    def __init__(self, df, img_dir, label_col="label", img_size=384, recipes=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label_col = label_col
        self.is_test = is_test

        self.recipes = recipes or []
        self.views_per_sample = max(1, len(self.recipes))

        self.val_tf = get_val_transform(img_size)

    def _load_rgb(self, img_id):
        path = os.path.join(self.img_dir, f"{img_id}")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __len__(self):
        if self.is_test:
            return len(self.df)
        return len(self.df) * self.views_per_sample

    def __getitem__(self, idx):
        if self.is_test:
            row = self.df.iloc[idx]
            img = self._load_rgb(row["ID"])
            x   = self.val_tf(image=img)["image"]
            return x, row["ID"]

        base_idx = idx // self.views_per_sample
        which    = idx %  self.views_per_sample
        row = self.df.iloc[base_idx]

        img = self._load_rgb(row["ID"])

        if self.recipes:
            x = self.recipes[which](image=img)["image"]
        else:
            x = self.val_tf(image=img)["image"]

        y = int(row[self.label_col])
        return x, y

    def _load_image(self, img_id):
        path = os.path.join(self.img_dir, f"{img_id}")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class DocDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['ID'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.is_test:
            return image, row['ID']
        else:
            target = int(row['target'])
            return image, target