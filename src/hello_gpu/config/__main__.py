import logging
from typing import TYPE_CHECKING

from src.hello_gpu.config.configuration import configuration

if TYPE_CHECKING:
    from src.hello_gpu.config.settings import Settings

logger = logging.getLogger(__name__)


def run() -> None:
    settings: Settings = configuration.settings
    logger.debug(settings.safe_model_dump())
    print(settings.model_dump())


if __name__ == "__main__":
    run()
