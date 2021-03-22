import os

import numpy as np
import torch
import pyprind

def get_dice_score(prediction, targets):
  prediction = (prediction > 0).float()
  return 2.0 * (prediction * targets).sum() / (prediction + targets).sum()

def train_model(model, criterion, optimizer, scheduler, num_epochs, train_dataloader, validation_dataloader, device):
  best_loss = float("inf")
  train_losses, validation_losses = [], []
  train_dices, validation_dices = [], []
  for epoch in range(num_epochs):
    print("Epoch: {}".format(epoch + 1))
    bar = pyprind.ProgBar(len(train_dataloader), title="Training Model")
    running_loss, running_dice = 0.0, 0.0
    for images, masks in train_dataloader:
      images, masks = images.to(device), masks.to(device)
      optimizer.zero_grad()

      outputs = torch.sigmoid(model(images))
      loss = criterion(outputs, masks)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * images.size(0)
      running_dice += get_dice_score(masks, outputs.detach()).item() * images.size(0)
      bar.update()

    train_losses.append(running_loss / float(len(train_dataloader.dataset)))
    train_dices.append(running_dice / float(len(train_dataloader.dataset)))

    with torch.no_grad():
      bar = pyprind.ProgBar(len(validation_dataloader), title="Evaluating Model")
      evaluating_loss, evaluating_dice = 0.0, 0.0
      for images, masks in validation_dataloader:
        images, masks = images.to(device), masks.to(device)
        outputs = torch.sigmoid(model(images))
        loss = criterion(outputs, masks)

        evaluating_loss += loss.item() * images.size(0)
        evaluating_dice += get_dice_score(masks, outputs.detach()).item() * images.size(0)
        bar.update()

      validation_losses.append(evaluating_loss / float(len(validation_dataloader.dataset)))
      validation_dices.append(evaluating_dice / float(len(validation_dataloader.dataset)))

      if validation_losses[-1] < best_loss:
        torch.save(model.state_dict(), os.path.join("models", "model_{}.pth".format(epoch + 1)))
        best_loss = validation_losses[-1]

    scheduler.step(validation_losses[-1])
    print("Train Loss: {:.3f}, Validation Loss: {:.3f}".format(train_losses[-1], validation_losses[-1]))
    print("Train Dices: {:.3f}, Validation Dices {:.3f}".format(train_dices[-1], validation_dices[-1]))
    print()

  history = {"train_losses": train_losses, "validation_losses": validation_losses, "train_dices": train_dices, "validation_dices": validation_dices}
  return history