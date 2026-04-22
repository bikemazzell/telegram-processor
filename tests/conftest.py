from __future__ import annotations
from pathlib import Path

import pytest

from telegram_processor import ChannelConfig, TelegramProcessor


@pytest.fixture
def processor(tmp_path: Path) -> TelegramProcessor:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel,password\nchan,@chan,secret\n", encoding="utf-8")

    return TelegramProcessor(
        input_file=input_file,
        start_date="1704067200",
        end_date="1704153600",
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
        settings_file=tmp_path / "missing-settings.json",
        process_only=False,
        auto_clean=True,
    )


@pytest.fixture
def channel(processor: TelegramProcessor) -> ChannelConfig:
    working_dir = processor.downloads_dir / "chan"
    working_dir.mkdir(parents=True)

    channel = ChannelConfig("chan", "@chan", ["secret"])
    channel.working_dir = working_dir
    return channel
