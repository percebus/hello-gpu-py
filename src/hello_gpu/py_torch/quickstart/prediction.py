import torch

from src.hello_gpu.py_torch.quickstart import models
from src.hello_gpu.py_torch.quickstart.data import test_data
from src.hello_gpu.py_torch.quickstart.device import device_name

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def run() -> None:
    models.load()
    models.model.eval()
    x = test_data[0][0]
    y = test_data[0][1]
    with torch.no_grad():
        x2 = x.to(device_name)
        pred = models.model(x2)
        first_prediction = pred[0].argmax(0)
        predicted = classes[first_prediction]
        actual = classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


if __name__ == "__main__":
    run()
