import torch

from src.hello_gpu.py_torch.quickstart.device import device_name
from src.hello_gpu.py_torch.quickstart.neural_network import neural_network

model = neural_network.to(device_name)
print(model)

default_path: str = "./data/model.pth"


def save(path: str = None):
    _path: str = path or default_path
    torch.save(model.state_dict(), _path)
    print("Saved PyTorch Model State to model.pth")


def load(path: str = None):
    _path: str = path or default_path
    model.load_state_dict(torch.load(_path, weights_only=True))
