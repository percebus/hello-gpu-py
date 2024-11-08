from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from src.hello_gpu.torch_.quickstart import testing, training

training_data: FashionMNIST = training.download_data()
test_data: FashionMNIST = testing.download_data()

batch_size = 64
training_dataloader: DataLoader =  DataLoader(training_data, batch_size=batch_size)
test_dataloader: DataLoader = DataLoader(test_data, batch_size=batch_size)
