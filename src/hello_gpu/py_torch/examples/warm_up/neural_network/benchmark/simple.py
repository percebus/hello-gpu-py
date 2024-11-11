import logging
import sys
import timeit

from src.hello_gpu.config.configuration import configuration
from src.hello_gpu.py_torch.examples.warm_up.neural_network import with_numpy, with_torch_cpu, with_torch_cuda

logger = configuration.get_logger(sys.modules[__name__].__name__)


def run() -> None:
    size = 2000
    samples = 10

    logger.info("With numpy...")
    result1 = timeit.timeit(lambda: with_numpy.create_neural_network(size), number=samples)
    logger.info(f" - result: {result1}")

    logger.info("With torch (cpu)...")
    result2 = timeit.timeit(lambda: with_torch_cpu.create_neural_network(size), number=samples)
    logger.info(f" - result: {result2}")

    logger.info("With torch (cuda)...")
    result3 = timeit.timeit(lambda: with_torch_cuda.create_neural_network(size), number=samples)
    logger.info(f" - result: {result3}")


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    run()
