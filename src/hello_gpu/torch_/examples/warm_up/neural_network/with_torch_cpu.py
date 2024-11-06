# -*- coding: utf-8 -*-

import math
import sys

import torch

from src.hello_gpu.config.configuration import configuration

logger = configuration.get_logger(sys.modules[__name__].__name__)


def create_neural_network(size: int = 2000) -> None:
    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    # Create random input and output data
    x = torch.linspace(-math.pi, math.pi, size, device=device, dtype=dtype)
    y = torch.sin(x)

    # Randomly initialize weights
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(size):
        # Forward pass: compute predicted y
        y_pred = a + b * x + c * x**2 + d * x**3

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
        if t % 100 == 99:
            logger.debug("%d%d", t, loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x**2).sum()
        grad_d = (grad_y_pred * x**3).sum()

        # Update weights using gradient descent
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    logger.debug("Result: y = %d + %d x + %d x^2 + %d x^3", a.item(), b.item(), c.item(), d.item())
