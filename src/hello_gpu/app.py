import os
import sys

from src.hello_gpu.config.configuration import configuration
from src.hello_gpu.cu_py.introduction.benchmark import simple as cu_py_benchmark
from src.hello_gpu.cuda_.overview import verification as cuda_verification
from src.hello_gpu.py_torch.examples.warm_up.neural_network.benchmark import simple as torch_benchmark

logger = configuration.get_logger(__name__)


def get_name() -> str:
    first_arg = sys.argv[0]
    base_name = os.path.basename(first_arg)
    app_parts = os.path.splitext(base_name)
    return app_parts[0]


def run_verifications() -> None:
    logger.info("Running verifications...")
    cuda_verification.verify()
    logger.info("Verifications completed.")


def run_benchmarks() -> None:
    logger.info("Running benchmarks...")
    cu_py_benchmark.run()
    torch_benchmark.run()
    logger.info("Benchmarks completed.")


def run() -> None:
    logger.info("Running application")
    run_verifications()
    run_benchmarks()
    logger.info("Application finished")
