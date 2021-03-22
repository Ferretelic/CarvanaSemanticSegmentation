import os
import random

import torch
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
from albumentations.pytorch import ToTensorV2

class CarvanaMaskingDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, image_size, mode):
    self.compose_transform(image_size)
    self.mode = mode
    np.random.seed(0)
    if mode != "test":
      image_path = os.path.join(dataset_path, "train")
      mask_path = os.path.join(dataset_path, "train_masks")
      train_indices, validation_indices = train_test_split(np.random.permutation(np.arange(len(os.listdir(image_path)))), random_state=0)
      if mode == "train":
        indices = train_indices
      else:
        indices = validation_indices
    else:
      image_path = os.path.join(dataset_path, "test")
      mask_path = os.path.join(dataset_path, "test_masks")
      indices = np.arange(len(os.listdir(image_path)))

    image_names = os.listdir(image_path)
    self.images = np.array([os.path.join(image_path, image_name) for image_name in image_names])[indices]
    self.masks = np.array([os.path.join(mask_path, "{}_mask.gif".format(os.path.splitext(image_name)[0])) for image_name in image_names])[indices]

  def __getitem__(self, index):
    image = np.asarray(Image.open(self.images[index]).convert("RGB"))
    mask = np.asarray(Image.open(self.masks[index]))
    if self.mode == "train":
      augmented = self.train_transform(image=image, mask=mask)
    else:
      augmented = self.validation_transform(image=image, mask=mask)
    image, mask = augmented["image"], augmented["mask"].unsqueeze(0).float()
    return image, mask

  def __len__(self):
    return len(self.images)

  def compose_transform(self, image_size):
    self.train_transform = Compose([
      Resize(*image_size, p=1.0),
      HorizontalFlip(p=0.5),
      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], p=1.0),
      ToTensorV2()
    ])
    self.validation_transform = Compose([
      Resize(*image_size, p=1.0),
      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], p=1.0),
      ToTensorV2()
    ])

def create_dataloaders(dataset_path, image_size, batch_size):
  train_dataloader = torch.utils.data.DataLoader(CarvanaMaskingDataset(dataset_path, image_size, "train"), batch_size=batch_size)
  validation_dataloader = torch.utils.data.DataLoader(CarvanaMaskingDataset(dataset_path, image_size, "validation"), batch_size=batch_size)

  return train_dataloader, validation_dataloader