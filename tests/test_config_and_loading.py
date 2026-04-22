from __future__ import annotations

import json
from pathlib import Path

import pytest

from telegram_processor import ChannelConfig, ProcessingError, Settings, TelegramProcessor


def test_channel_config_trims_values_and_passwords() -> None:
    channel = ChannelConfig(" chan ", " @chan ", [" secret ", "", "   ", None])  # type: ignore[list-item]

    assert channel.name == "chan"
    assert channel.channel == "@chan"
    assert channel.passwords == ["secret"]
    assert channel.has_passwords is True


def test_settings_missing_file_uses_defaults(tmp_path: Path) -> None:
    settings = Settings(tmp_path / "missing.json")

    assert settings.get("tdl", "max_parallel_downloads") == 1
    assert settings.get("archive", "supported_extensions") == [".zip", ".rar", ".7z"]


def test_settings_invalid_json_raises_processing_error(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ProcessingError):
        Settings(settings_file)


def test_prepare_extensions_returns_lower_and_upper_variants(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({}), encoding="utf-8")
    settings = Settings(settings_file)

    result = settings.prepare_extensions(["zip", ".rar"])

    assert set(result) == {"zip", "ZIP", "rar", "RAR"}


def test_load_channels_supports_multiple_password_columns(tmp_path: Path) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text(
        "name,channel,password1,password2\nchan,@chan,one,two\n",
        encoding="utf-8",
    )

    processor = TelegramProcessor(
        input_file=input_file,
        start_date="1704067200",
        end_date="1704153600",
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
        settings_file=tmp_path / "settings.json",
    )

    processor.load_channels()

    assert len(processor.channels) == 1
    assert processor.channels[0].passwords == ["one", "two"]


def test_load_channels_rejects_invalid_header(tmp_path: Path) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("wrong,channel\nchan,@chan\n", encoding="utf-8")

    processor = TelegramProcessor(
        input_file=input_file,
        start_date="1704067200",
        end_date="1704153600",
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
        settings_file=tmp_path / "settings.json",
    )

    with pytest.raises(ProcessingError):
        processor.load_channels()
