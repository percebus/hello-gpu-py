import logging
from dataclasses import dataclass, field

from src.hello_gpu.config.logs import Logging
from src.hello_gpu.config.settings import Settings


@dataclass
class Configuration:
    settings: Settings = field(default_factory=Settings)

    logging: Logging = field(default_factory=Logging)

    def configure_logging(self) -> None:
        self.logging.configure(self.settings.logging_config)
        if self.settings.debug:
            self.logging.logger.setLevel(logging.DEBUG)

    def __post_init__(self) -> None:
        self.configure_logging()
        logging.debug("Configuration: initializing...")


configuration = Configuration()
