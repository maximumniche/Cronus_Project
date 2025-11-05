"""app-pytorch: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


# TODO: REPLACE WITH OUR MODEL
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

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


fds = None  # Cache FederatedDataset

#pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#pytorch_transforms = ToTensor()
pytorch_transforms = Compose([ToTensor(), nn.Flatten(start_dim=0)]) # Convert 28x28 MNIST array to tensor and flatten it to 1 dim to feed in


def apply_transforms(batch):
    #Apply transforms to the partition from FederatedDataset.
    #print(batch)
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    #for i in range(len(batch["image"])):
    #    print("ARRAY SHAPE\n\n\n\n\n\n\n\n")
    #    print(batch["image"][i].shape)
    return batch

def load_data(partition_id: int, num_partitions: int):

    """Load partition cifar10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    #print(trainLoader)
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("ylecun/mnist", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=32)
    #return DataLoader(test_dataset, batch_size=32)

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            print("outputs.shape:", outputs.shape)
            print("labels.shape:", labels.shape)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy