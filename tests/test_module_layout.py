from __future__ import annotations

from pathlib import Path

from telegram_processor import ChannelConfig, Settings, SystemOps, TelegramProcessor, setup_logging
from telegram_processor_lib.base import BaseProcessorMixin
from telegram_processor_lib.config import ChannelConfig as PackageChannelConfig
from telegram_processor_lib.download import DownloadMixin
from telegram_processor_lib.config import Settings as PackageSettings
from telegram_processor_lib.logging_utils import setup_logging as package_setup_logging
from telegram_processor_lib.processing_steps import ProcessingStepsMixin
from telegram_processor_lib.processor import TelegramProcessor as PackageTelegramProcessor
from telegram_processor_lib.system_ops import SystemOps as PackageSystemOps
from telegram_processor_lib.workflow import WorkflowMixin


def test_public_api_is_reexported_from_package() -> None:
    assert ChannelConfig is PackageChannelConfig
    assert Settings is PackageSettings
    assert SystemOps is PackageSystemOps
    assert TelegramProcessor is PackageTelegramProcessor
    assert setup_logging is package_setup_logging


def test_package_processor_still_constructs_with_existing_cli_paths(tmp_path: Path) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nchan,@chan\n", encoding="utf-8")

    processor = PackageTelegramProcessor(
        input_file=input_file,
        start_date="1704067200",
        end_date="1704153600",
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
        settings_file=tmp_path / "settings.json",
    )

    assert processor.input_file == input_file


def test_package_processor_is_composed_from_smaller_mixins() -> None:
    assert issubclass(PackageTelegramProcessor, BaseProcessorMixin)
    assert issubclass(PackageTelegramProcessor, DownloadMixin)
    assert issubclass(PackageTelegramProcessor, ProcessingStepsMixin)
    assert issubclass(PackageTelegramProcessor, WorkflowMixin)
