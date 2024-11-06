import json
import logging
import logging.config
from dataclasses import dataclass, field
from logging import Logger
from typing import Any, Optional


@dataclass
class Logging:
    config: Optional[dict[str, Any]] = field(init=False, default=None)

    def create_logger(self, name: str = __name__) -> Logger:
        logger: Logger = logging.getLogger(name)
        # logger.propagate = False # TODO?
        if self.config:
            logger.debug("Configuring logger with dictConfig")
            logging.config.dictConfig(self.config)
        else:
            logging.basicConfig(level=logging.INFO)
            logger.debug("Configuring logger with basicConfig")

        return logger

    # TODO change for logging.config
    def load_configuration(self, path: str = None) -> None:
        logging.debug("Loading logging configuration from file %s", path)
        _path = path or "data/config/logging.json"
        with open(_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
