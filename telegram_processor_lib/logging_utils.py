from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional


LOGGER_NAME = "telegram_processor"
logger = logging.getLogger(LOGGER_NAME)


def setup_logging(verbose: bool = False, settings: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Setup logging with file rotation and console output."""
    configured_logger = logging.getLogger(LOGGER_NAME)
    configured_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    for handler in list(configured_logger.handlers):
        configured_logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    if settings:
        max_size = settings.get("logging", {}).get("max_file_size_mb", 10) * 1024 * 1024
        backup_count = settings.get("logging", {}).get("backup_count", 5)
    else:
        max_size = 10 * 1024 * 1024
        backup_count = 5

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "telegram_processor.log",
        maxBytes=max_size,
        backupCount=backup_count,
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    configured_logger.addHandler(console_handler)
    configured_logger.addHandler(file_handler)

    return configured_logger
