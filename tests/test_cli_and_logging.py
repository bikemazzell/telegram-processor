from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest

import telegram_processor
from telegram_processor_lib import system_ops as system_ops_module


def test_setup_logging_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    logger = logging.getLogger(telegram_processor.__name__)
    logger.handlers.clear()

    first = telegram_processor.setup_logging()
    first_handler_count = len(first.handlers)

    second = telegram_processor.setup_logging()

    assert first is second
    assert len(second.handlers) == first_handler_count


def test_main_process_only_does_not_require_tdl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nchan,@chan\n", encoding="utf-8")

    parsed_args = argparse.Namespace(
        input=str(input_file),
        start="01-01-2026",
        end="02-01-2026",
        output_dir=str(tmp_path / "output"),
        download_dir=str(tmp_path / "downloads"),
        settings=None,
        verbose=False,
        process_only=True,
        auto_clean=True,
    )

    monkeypatch.setattr(telegram_processor.argparse.ArgumentParser, "parse_args", lambda self: parsed_args)
    monkeypatch.setattr(telegram_processor, "setup_logging", lambda verbose, settings=None: logging.getLogger("test"))
    monkeypatch.setattr(telegram_processor.TelegramProcessor, "process", lambda self: None)
    monkeypatch.setattr(
        system_ops_module.SystemOps,
        "which",
        lambda self, tool: None if tool == "tdl" else f"/usr/bin/{tool}",
    )

    telegram_processor.main()
