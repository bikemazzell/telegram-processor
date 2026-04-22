from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging_utils import logger


class ProcessingError(Exception):
    """Base exception for processing errors."""


class ChannelConfig:
    """Configuration for a Telegram channel."""

    def __init__(
        self,
        name: str,
        channel: str,
        passwords: Optional[List[str]] = None,
        password_source: Optional[str] = None,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("Channel name cannot be empty")
        if not channel or not channel.strip():
            raise ValueError("Channel identifier cannot be empty")

        self.name: str = name.strip()
        self.channel: str = channel.strip()
        self.password_source: Optional[str] = password_source.strip() if password_source and password_source.strip() else None
        self.passwords: List[str] = [p.strip() for p in passwords if p and p.strip()] if passwords else []
        self.working_dir: Optional[Path] = None

    @property
    def has_passwords(self) -> bool:
        return bool(self.passwords) and not self.uses_post_passwords

    @property
    def uses_post_passwords(self) -> bool:
        return self.password_source == "password_in_post"


class Settings:
    """Configuration settings loaded from JSON file."""

    DEFAULT_SETTINGS_FILE = "settings.json"

    def __init__(self, settings_file: Optional[str | Path] = None) -> None:
        self.settings_file = Path(settings_file or self.DEFAULT_SETTINGS_FILE)
        self.settings: Dict[str, Any] = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        try:
            with open(self.settings_file, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            logger.warning("Settings file not found: %s, using defaults", self.settings_file)
            return self._default_settings()
        except json.JSONDecodeError as exc:
            raise ProcessingError(f"Invalid JSON in settings file: {exc}")
        except Exception as exc:
            raise ProcessingError(f"Failed to load settings: {exc}")

    def _default_settings(self) -> Dict[str, Any]:
        return {
            "stealer_log_processor": {"path": "./stealer-log-processor/main.py"},
            "tdl": {
                "max_parallel_downloads": 1,
                "reconnect_timeout": 0,
                "download_timeout": 7200,
                "export_channel_threads": 4,
                "bandwidth_limit": 0,
                "chunk_size": 128,
                "excluded_extensions": ["jpg", "gif", "png", "webp", "webm", "mp4"],
                "included_extensions": ["zip", "rar", "7z", "txt", "csv"],
                "max_retries": 3,
                "retry_delay": 5,
            },
            "sort": {
                "memory_percent": 30,
                "max_parallel": 16,
                "temp_dir": "/tmp",
                "chunk_size": 1000000,
            },
            "archive": {
                "extract_patterns": ["*.txt", "*.csv", "*pass*", "*auto*"],
                "supported_extensions": [".zip", ".rar", ".7z"],
                "extract_timeout": 300,
                "max_retries": 2,
                "retry_delay": 2,
            },
            "processing": {
                "max_workers": None,
                "checkpoint_interval": 100,
                "min_file_size_bytes": 0,
                "min_result_file_size_bytes": 1,
            },
            "logging": {
                "max_file_size_mb": 10,
                "backup_count": 5,
                "temp_cleanup_patterns": ["tdl-export*.json", "*.temp", "sort*"],
            },
            "subprocess": {
                "default_timeout": 300,
                "terminate_timeout": 5,
            },
        }

    def get(self, *keys, default=None):
        try:
            current = self.settings
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def prepare_extensions(self, extensions: List[str]) -> List[str]:
        result = set()
        for ext in extensions:
            ext = ext.lower().strip()
            if ext:
                if ext.startswith("."):
                    ext = ext[1:]
                result.add(ext)
                upper_ext = ext.upper()
                if upper_ext != ext:
                    result.add(upper_ext)
        return list(result)
