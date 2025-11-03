"""app-pytorch: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, datasets

# TODO: REPLACE WITH OUR MODEL
class MnistModel(nn.Module):
  def __init__(self) -> None:
      super().__init__()
      self.lin1 = nn.Linear(784, 256)
      self.lin2 = nn.Linear(256, 64)
      self.lin3 = nn.Linear(64, 10)

  def forward(self, X):
      x1 = F.relu(self.lin1(X))
      x2 = F.relu(self.lin2(x1))
      x3 = F.relu(self.lin3(x2))
      return x3

def load_local():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    return DataLoader(mnist_trainset, batch_size=32)


# TODO REPLACE WITH OUT TRAIN/TEST FUNCTIONS

 # Fit function
def fit(self, X, y, optimizer, loss_fn, epochs):

    for epoch in range(epochs):

        ypred = self.forward(X)
        loss = loss_fn(ypred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy