import pickle

import segmentation_models_pytorch
import torch

from dataset import create_dataloaders
from history import plot_history
from utils import train_model

dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/CarvanaImageMaskingDataset"
image_size = (128, 128)
batch_size = 64
device = torch.device("cuda")
train_dataloader, validation_dataloader = create_dataloaders(dataset_path, image_size, batch_size)

num_epochs = 10

model = segmentation_models_pytorch.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)

history = train_model(model, criterion, optimizer, scheduler, num_epochs, train_dataloader, validation_dataloader, device)
plot_history(history, num_epochs)

with open("./histories/history.pkl", "wb") as f:
  pickle.dump(history, f)
