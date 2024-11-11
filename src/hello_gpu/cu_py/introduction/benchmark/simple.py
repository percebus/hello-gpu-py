import logging
import sys
import timeit

import cupy
import numpy

from src.hello_gpu.config.configuration import configuration

logger = configuration.get_logger(sys.modules[__name__].__name__)


def run() -> None:
    x = 999
    y = 99

    # load a matrix to global memory
    array_cpu = numpy.random.randint(0, 255, size=(x, y))

    # benchmark matrix addition on CPU by using a NumPy addition function
    logger.info("With numpy @ CPU")
    cpu_time = timeit.timeit(lambda: numpy.add(array_cpu, 999))
    logger.info(" - result: %f", cpu_time)

    # load the same matrix to GPU memory
    array_gpu = cupy.asarray(array_cpu)

    # benchmark matrix addition on GPU by using CuPy addition function
    logger.info("With cupy @ GPU")
    # you need to run a pilot iteration on a GPU first to compile and cache the function kernel on a GPU
    cupy.add(array_gpu, 1)
    gpu_time = timeit.timeit(lambda: cupy.add(array_gpu, 999))
    logger.info(" - result: %f", gpu_time)

    # determine how much is GPU faster
    # faster_processor = (gpu_time - cpu_time) / gpu_time * 100
    # logger.info(f"CPU time: {cpu_time} seconds\nGPU time: {gpu_time} seconds.\nGPU was {faster_processor} percent faster")


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    run()
