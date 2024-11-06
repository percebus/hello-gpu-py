import logging
from dataclasses import dataclass, field
from logging import Logger

from src.hello_gpu.config.logs import Logging
from src.hello_gpu.config.settings import Settings


@dataclass
class Configuration:
    settings: Settings = field(default_factory=Settings)

    logging: Logging = field(default_factory=Logging)

    def __post_init__(self) -> None:
        # self.logging.load_configuration(self.settings.logging_config) # FIXME bad JSON?
        logging.debug("Configuration: initializing...")

    def get_logger(self, name: str = __name__) -> Logger:
        logger = self.logging.create_logger(name)
        if self.settings.debug:
            logger.setLevel(logging.DEBUG)

        return logger


configuration = Configuration()
