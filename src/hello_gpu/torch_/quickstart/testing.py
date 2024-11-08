import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from src.hello_gpu.torch_.quickstart.device import device_name


def run(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device_name), y.to(device_name)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def download_data() -> FashionMNIST:
    # Download test data from open datasets.
    return FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
