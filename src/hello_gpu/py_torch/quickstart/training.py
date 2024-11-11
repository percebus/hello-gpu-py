import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from src.hello_gpu.py_torch.quickstart.device import device_name
from src.hello_gpu.py_torch.quickstart.models import model

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def run(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device_name), y.to(device_name)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def download_data() -> FashionMNIST:
    # Download training data from open datasets.
    return FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
