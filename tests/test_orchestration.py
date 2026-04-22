from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import telegram_processor
from telegram_processor_lib import workflow as workflow_module
from telegram_processor import ChannelConfig, TelegramProcessor


@dataclass(frozen=True)
class FakeFuture:
    result_value: tuple[ChannelConfig, bool]

    def result(self) -> tuple[ChannelConfig, bool]:
        return self.result_value


class FakeExecutor:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers
        self.futures: list[FakeFuture] = []

    def __enter__(self) -> "FakeExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def submit(self, fn, channel):  # noqa: ANN001
        future = FakeFuture(fn(channel))
        self.futures.append(future)
        return future


class FakeSystem:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.removed: list[Path] = []

    def prompt(self, message: str) -> str:
        self.prompts.append(message)
        return "y"

    def rmtree(self, path: Path) -> None:
        self.removed.append(path)


class FailingExecutor:
    def __init__(self, max_workers: int) -> None:  # noqa: ARG002
        raise AssertionError("executor should not be created")


def test_process_only_uses_existing_dirs_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nchan,@chan\n", encoding="utf-8")

    processor = TelegramProcessor(
        input_file=input_file,
        start_date="1704067200",
        end_date="1704153600",
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
        settings_file=tmp_path / "settings.json",
        process_only=True,
        auto_clean=True,
        process_executor_cls=FakeExecutor,
    )

    working_dir = processor.downloads_dir / "chan"
    working_dir.mkdir(parents=True)
    (working_dir / "existing.txt").write_text("data\n", encoding="utf-8")

    processed = []
    removed = []

    def fake_process_channel_files(channel: ChannelConfig) -> tuple[ChannelConfig, bool]:
        processed.append(channel.name)
        return channel, True

    monkeypatch.setattr(processor, "process_channel_files", fake_process_channel_files)
    monkeypatch.setattr(workflow_module, "as_completed", lambda futures: futures)
    monkeypatch.setattr(processor, "remove_tree", lambda path: removed.append(path))

    processor.process()

    assert processed == ["chan"]
    assert removed == [working_dir]


def test_process_uses_injected_prompt_and_cleanup_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nchan,@chan\n", encoding="utf-8")

    fake_system = FakeSystem()
    processor = TelegramProcessor(
        input_file=input_file,
        start_date="1704067200",
        end_date="1704153600",
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
        settings_file=tmp_path / "settings.json",
        process_only=True,
        auto_clean=False,
        process_executor_cls=FakeExecutor,
        system_ops=fake_system,
    )

    working_dir = processor.downloads_dir / "chan"
    working_dir.mkdir(parents=True)
    (working_dir / "existing.txt").write_text("data\n", encoding="utf-8")

    monkeypatch.setattr(processor, "process_channel_files", lambda channel: (channel, True))
    monkeypatch.setattr(workflow_module, "as_completed", lambda futures: futures)

    processor.process()

    assert fake_system.prompts == ["Do you want to clean up 1 processed directories? (y/n): "]
    assert fake_system.removed == [working_dir]


def test_main_invalid_date_exits_with_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nchan,@chan\n", encoding="utf-8")

    parsed_args = type(
        "Args",
        (),
        {
            "input": str(input_file),
            "start": "bad-date",
            "end": "02-01-2026",
            "output_dir": str(tmp_path / "output"),
            "download_dir": str(tmp_path / "downloads"),
            "settings": None,
            "verbose": False,
            "process_only": True,
            "auto_clean": True,
        },
    )()

    monkeypatch.setattr(telegram_processor.argparse.ArgumentParser, "parse_args", lambda self: parsed_args)
    monkeypatch.setattr(telegram_processor, "setup_logging", lambda verbose, settings=None: __import__("logging").getLogger("test"))

    with pytest.raises(SystemExit) as exc_info:
        telegram_processor.main()

    assert exc_info.value.code == 1


def test_process_download_mode_runs_download_dedup_process_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nfirst,@first\nsecond,@second\n", encoding="utf-8")

    processor = TelegramProcessor(
        input_file=input_file,
        start_date="1704067200",
        end_date="1704153600",
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
        settings_file=tmp_path / "settings.json",
        process_only=False,
        auto_clean=True,
        process_executor_cls=FakeExecutor,
    )

    first_dir = processor.downloads_dir / "first"
    first_dir.mkdir(parents=True)
    (first_dir / "file.txt").write_text("data\n", encoding="utf-8")

    download_calls: list[str] = []
    dedup_targets: list[Path] = []
    processed: list[str] = []
    removed: list[Path] = []

    def fake_download(channel: ChannelConfig) -> tuple[ChannelConfig, bool, str | None]:
        download_calls.append(channel.name)
        if channel.name == "first":
            channel.working_dir = first_dir
            return channel, True, None
        return channel, False, "no files"

    def fake_process_channel(channel: ChannelConfig) -> tuple[ChannelConfig, bool]:
        processed.append(channel.name)
        return channel, True

    monkeypatch.setattr(processor, "download_channel_wrapper", fake_download)
    monkeypatch.setattr(processor, "deduplicate", lambda path: dedup_targets.append(path))
    monkeypatch.setattr(processor, "process_channel_files", fake_process_channel)
    monkeypatch.setattr(processor, "remove_tree", lambda path: removed.append(path))
    monkeypatch.setattr(workflow_module, "as_completed", lambda futures: futures)

    processor.process()

    assert download_calls == ["first", "second"]
    assert dedup_targets == [processor.downloads_dir]
    assert processed == ["first"]
    assert removed == [first_dir]


def test_process_process_only_with_no_existing_dirs_exits_before_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nfirst,@first\n", encoding="utf-8")

    processor = TelegramProcessor(
        input_file=input_file,
        start_date="1704067200",
        end_date="1704153600",
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
        settings_file=tmp_path / "settings.json",
        process_only=True,
        auto_clean=True,
        process_executor_cls=FailingExecutor,
    )

    monkeypatch.setattr(workflow_module, "as_completed", lambda futures: futures)

    processor.process()


def test_main_success_runs_full_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nchan,@chan\n", encoding="utf-8")
    calls: list[str] = []

    parsed_args = type(
        "Args",
        (),
        {
            "input": str(input_file),
            "start": "01-01-2026",
            "end": "02-01-2026",
            "output_dir": str(tmp_path / "output"),
            "download_dir": str(tmp_path / "downloads"),
            "settings": None,
            "verbose": False,
            "process_only": True,
            "auto_clean": True,
        },
    )()

    monkeypatch.setattr(telegram_processor.argparse.ArgumentParser, "parse_args", lambda self: parsed_args)
    monkeypatch.setattr(telegram_processor, "setup_logging", lambda verbose, settings=None: __import__("logging").getLogger("test"))
    monkeypatch.setattr(telegram_processor.TelegramProcessor, "check_dependencies", lambda self: calls.append("check"))
    monkeypatch.setattr(telegram_processor.TelegramProcessor, "process", lambda self: calls.append("process"))
    monkeypatch.setattr(telegram_processor.TelegramProcessor, "cleanup_temp_files", lambda self: calls.append("cleanup"))

    telegram_processor.main()

    assert calls == ["check", "process", "cleanup"]


def test_main_processing_error_still_cleans_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nchan,@chan\n", encoding="utf-8")
    calls: list[str] = []

    parsed_args = type(
        "Args",
        (),
        {
            "input": str(input_file),
            "start": "01-01-2026",
            "end": "02-01-2026",
            "output_dir": str(tmp_path / "output"),
            "download_dir": str(tmp_path / "downloads"),
            "settings": None,
            "verbose": False,
            "process_only": True,
            "auto_clean": True,
        },
    )()

    monkeypatch.setattr(telegram_processor.argparse.ArgumentParser, "parse_args", lambda self: parsed_args)
    monkeypatch.setattr(telegram_processor, "setup_logging", lambda verbose, settings=None: __import__("logging").getLogger("test"))

    def fake_check(self) -> None:
        calls.append("check")
        raise telegram_processor.ProcessingError("boom")

    monkeypatch.setattr(telegram_processor.TelegramProcessor, "check_dependencies", fake_check)
    monkeypatch.setattr(telegram_processor.TelegramProcessor, "cleanup_temp_files", lambda self: calls.append("cleanup"))

    with pytest.raises(SystemExit) as exc_info:
        telegram_processor.main()

    assert exc_info.value.code == 1
    assert calls == ["check", "cleanup"]


def test_process_download_mode_with_no_successful_downloads_exits_before_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nfirst,@first\nsecond,@second\n", encoding="utf-8")

    processor = TelegramProcessor(
        input_file=input_file,
        start_date="1704067200",
        end_date="1704153600",
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
        settings_file=tmp_path / "settings.json",
        process_only=False,
        auto_clean=True,
        process_executor_cls=FailingExecutor,
    )

    dedup_targets: list[Path] = []

    monkeypatch.setattr(processor, "download_channel_wrapper", lambda channel: (channel, False, "skip"))
    monkeypatch.setattr(processor, "deduplicate", lambda path: dedup_targets.append(path))

    processor.process()

    assert dedup_targets == []


def test_process_declined_cleanup_preserves_processed_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nchan,@chan\n", encoding="utf-8")

    fake_system = FakeSystem()
    fake_system.prompt = lambda message: "n"  # type: ignore[method-assign]
    processor = TelegramProcessor(
        input_file=input_file,
        start_date="1704067200",
        end_date="1704153600",
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
        settings_file=tmp_path / "settings.json",
        process_only=True,
        auto_clean=False,
        process_executor_cls=FakeExecutor,
        system_ops=fake_system,
    )

    working_dir = processor.downloads_dir / "chan"
    working_dir.mkdir(parents=True)
    (working_dir / "existing.txt").write_text("data\n", encoding="utf-8")

    monkeypatch.setattr(processor, "process_channel_files", lambda channel: (channel, True))
    monkeypatch.setattr(workflow_module, "as_completed", lambda futures: futures)

    processor.process()

    assert fake_system.removed == []
    assert working_dir.exists()


def test_process_removes_checkpoint_after_mixed_processing_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\none,@one\ntwo,@two\n", encoding="utf-8")

    processor = TelegramProcessor(
        input_file=input_file,
        start_date="1704067200",
        end_date="1704153600",
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
        settings_file=tmp_path / "settings.json",
        process_only=True,
        auto_clean=True,
        process_executor_cls=FakeExecutor,
    )

    one_dir = processor.downloads_dir / "one"
    two_dir = processor.downloads_dir / "two"
    one_dir.mkdir(parents=True)
    two_dir.mkdir(parents=True)
    (one_dir / "a.txt").write_text("1\n", encoding="utf-8")
    (two_dir / "b.txt").write_text("2\n", encoding="utf-8")
    processor.checkpoint_file.write_text('{"one": true}', encoding="utf-8")

    removed: list[Path] = []

    def fake_process_channel(channel: ChannelConfig) -> tuple[ChannelConfig, bool]:
        return channel, channel.name == "one"

    monkeypatch.setattr(processor, "process_channel_files", fake_process_channel)
    monkeypatch.setattr(processor, "remove_tree", lambda path: removed.append(path))
    monkeypatch.setattr(workflow_module, "as_completed", lambda futures: futures)

    processor.process()

    assert removed == [one_dir]
    assert not processor.checkpoint_file.exists()
