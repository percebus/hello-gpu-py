from src.hello_gpu.cuda_.overview.verification import verify as verify_cuda
from src.hello_gpu.torch_.verification import verify_all as verify_torch

functions = [
    verify_cuda,
    verify_torch,
]

def verify_all() -> None:
    for fn in functions:
        fn()
