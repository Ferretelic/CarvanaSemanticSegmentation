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
model.load_state_dict(torch.load("./models/model_10.pth"))

dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/CarvanaImageMaskingDataset"
image_size = (128, 128)
image_path = os.path.join(dataset_path, "test")

num_images = 3
transform = Compose([
      Resize(*image_size, p=1.0),
      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], p=1.0),
      ToTensorV2()
])
figure, axes = plt.subplots(num_images, 2, figsize=(5, 7.5))

for index, ((ax1, ax2), image_name) in enumerate(zip(axes, np.random.choice(os.listdir(image_path), num_images))):
      image = np.asarray(Image.open(os.path.join(image_path, image_name)).resize(image_size).convert("RGB"))
      mask = torch.sigmoid(model(transform(image=image)["image"].unsqueeze(0))).squeeze().detach().numpy()

      ax1.imshow(image)
      ax1.set_xticks([])
      ax1.set_yticks([])
      ax2.imshow(mask, cmap="gray")
      ax2.set_xticks([])
      ax2.set_yticks([])

      if index == 0:
            ax1.set_title("image")
            ax2.set_title("mask")

figure.tight_layout()
figure.savefig("./predicted_sample.png")