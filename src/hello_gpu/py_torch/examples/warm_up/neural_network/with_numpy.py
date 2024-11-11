# -*- coding: utf-8 -*-

import math
import sys

import numpy as np

from src.hello_gpu.config.configuration import configuration

logger = configuration.get_logger(sys.modules[__name__].__name__)


def create_neural_network(size: int = 2000) -> None:
    # Create random input and output data
    x = np.linspace(-math.pi, math.pi, size)
    y = np.sin(x)

    # Randomly initialize weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()

    learning_rate = 1e-6
    for t in range(size):
        # Forward pass: compute predicted y
        # y = a + b x + c x^2 + d x^3
        y_pred = a + b * x + c * x**2 + d * x**3

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        if t % 100 == 99:
            logger.debug("%d%d", t, loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x**2).sum()
        grad_d = (grad_y_pred * x**3).sum()

        # Update weights
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    logger.debug("Result: y = %d + %d x + %d x^2 + %d x^3", a, b, c, d)
