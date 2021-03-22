import matplotlib.pyplot as plt
import seaborn as sns

def plot_history(history, num_epochs):
  train_losses, validation_losses = history["train_losses"], history["validation_losses"]
  train_dices, validation_dices = history["train_dices"], history["validation_dices"]

  plt.figure()
  plt.title("Loss")
  sns.lineplot(x=range(num_epochs), y=train_losses, legend="brief", label="train loss")
  sns.lineplot(x=range(num_epochs), y=validation_losses, legend="brief", label="validation loss")
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.savefig("./histories/loss.png")

  plt.figure()
  plt.title("Dice")
  sns.lineplot(x=range(num_epochs), y=train_dices, legend="brief", label="train dice")
  sns.lineplot(x=range(num_epochs), y=validation_dices, legend="brief", label="validation dice")
  plt.xlabel("epoch")
  plt.ylabel("dice")
  plt.savefig("./histories/dice.png")
