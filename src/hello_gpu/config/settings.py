from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # NOTE: os.environ takes precedence over .env files
    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
        env_file=(  # Each file overrides the previous one
            ".env",  # lowest precedence
            ".env.local",  # highest precedence
        ),
    )

    debug: Optional[bool] = Field(default=False)

    logging_config: Optional[str] = Field(default="data/config/logging.json")

    def safe_model_dump(self) -> Optional[dict[str, Any]]:
        return self.model_dump() if self.debug else None
