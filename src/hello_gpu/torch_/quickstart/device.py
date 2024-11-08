import torch

# Get cpu, gpu or mps device for training.
device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device_name} device")
