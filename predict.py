import os

from PIL import Image
import torch
import torchvision
import segmentation_models_pytorch
import numpy as np
import matplotlib.pyplot as plt
from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
from albumentations.pytorch import ToTensorV2

model = segmentation_models_pytorch.Unet("resnet18", encoder_weights=None, classes=1, activation=None)
model.load_state_dict(torch.load("./models/model_9.pth"))

dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/CarvanaImageMaskingDataset"
image_size = (128, 128)

image_path = os.path.join(dataset_path, "test")
image = np.asarray(Image.open(os.path.join(image_path, os.listdir(image_path)[1009])).convert("RGB"))
transform = Compose([
      Resize(*image_size, p=1.0),
      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], p=1.0),
      ToTensorV2()
])

mask = torch.sigmoid(model(transform(image=image)["image"].unsqueeze(0))).squeeze().detach().numpy()
image = np.asarray(Image.open(os.path.join(image_path, os.listdir(image_path)[1009])).resize((128, 128)))
figure, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image)
ax1.set_title("image")
ax2.imshow(mask, cmap="gray")
ax2.set_title("mask")
figure.savefig("./sample.png")