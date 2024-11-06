import torch


# SRC: https://pytorch.org/get-started/locally/#windows-pip
def verify_tensor() -> None:
    # The output should be something similar to:
    #
    # tensor([
    #   [0.3380, 0.3845, 0.3217],
    #   [0.8337, 0.9050, 0.2650],
    #   [0.2979, 0.7141, 0.9069],
    #   [0.1449, 0.1132, 0.1375],
    #   [0.4675, 0.3947, 0.1426]
    # ])
    x = torch.rand(5, 3)
    print(x)
    print('tensor is available')


def verify_cuda() -> bool:
    print(torch.version.cuda)
    is_available = torch.cuda.is_available()
    if not is_available:
        raise ValueError('CUDA is not available')

    print('CUDA is available')
    print(torch.cuda.get_device_properties(0).total_memory)
    return is_available


def verify_all() -> None:
    verify_tensor()
    verify_cuda()
