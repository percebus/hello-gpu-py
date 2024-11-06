import json
import logging
import logging.config
from dataclasses import dataclass, field
from logging import Logger
from typing import Any, Optional


@dataclass
class Logging:
    config: Optional[dict[str, Any]] = field(init=False, default=None)

    logger: Logger = field(init=False)

    def get_logger(self, name: str = __name__) -> Logger:
        logger: Logger = logging.getLogger(name)
        logger.propagate = False
        if self.config:
            logging.config.dictConfig(self.config)

        return logger

    def configure(self, path: str = None) -> None:
        _path = path or "data/config/logging.json"
        with open(_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def __post_init__(self) -> None:
        self.logger = self.get_logger()
