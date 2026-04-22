from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from telegram_processor_lib import download as download_module
from telegram_processor import ChannelConfig, TelegramProcessor


def test_parse_export_file_returns_only_messages_with_files(processor: TelegramProcessor, channel: ChannelConfig) -> None:
    export_file = processor.downloads_dir / channel.name / processor.EXPORT_FILE
    export_file.parent.mkdir(parents=True, exist_ok=True)
    export_file.write_text(
        json.dumps(
            {
                "messages": [
                    {"type": "message", "file": "one.zip"},
                    {"type": "message"},
                    {"type": "service", "file": "ignored.zip"},
                    {"type": "message", "file": "two.txt"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert processor.parse_export_file(channel) == ["one.zip", "two.txt"]


def test_get_archive_files_respects_supported_extensions(processor: TelegramProcessor, channel: ChannelConfig) -> None:
    (channel.working_dir / "one.zip").write_text("a", encoding="utf-8")
    (channel.working_dir / "two.RAR").write_text("a", encoding="utf-8")
    (channel.working_dir / "three.txt").write_text("a", encoding="utf-8")

    files = processor.get_archive_files(channel.working_dir)

    assert {file.name for file in files} == {"one.zip", "two.RAR"}


def test_retry_operation_retries_until_success(processor: TelegramProcessor, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}
    monkeypatch.setattr(download_module.time, "sleep", lambda _: None)

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("boom")
        return "ok"

    assert processor.retry_operation(flaky, max_retries=3, retry_delay=1) == "ok"
    assert attempts["count"] == 3


def test_process_text_files_streaming_combines_unique_lines(
    processor: TelegramProcessor,
    channel: ChannelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (channel.working_dir / "first.txt").write_text("b\na\n", encoding="utf-8")
    (channel.working_dir / "second.txt").write_text("a\nc\n", encoding="utf-8")

    moved = {}

    def fake_run(cmd, check, cwd, env):  # noqa: ANN001
        assert cmd[0] == "sort"
        output_name = cmd[cmd.index("-o") + 1]
        input_file = Path(cmd[-1])
        lines = sorted(set(input_file.read_text(encoding="utf-8").splitlines()))
        (Path(cwd) / output_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    def fake_move(src: Path, dst: Path) -> None:
        moved["src"] = src
        moved["dst"] = dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        src.unlink()

    monkeypatch.setattr(processor.system, "run", fake_run)
    monkeypatch.setattr(processor, "safe_move", fake_move)

    assert processor.process_text_files_streaming(channel) is True
    assert moved["dst"] == processor.output_dir / f"{channel.name}-{processor.date_suffix}-combo.csv"
    assert moved["dst"].read_text(encoding="utf-8") == "a\nb\nc\n"


def test_process_text_files_streaming_cleans_up_on_sort_failure(
    processor: TelegramProcessor,
    channel: ChannelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (channel.working_dir / "first.txt").write_text("b\na\n", encoding="utf-8")

    def fake_run(cmd, check, cwd, env):  # noqa: ANN001
        output_name = cmd[cmd.index("-o") + 1]
        (Path(cwd) / output_name).write_text("partial\n", encoding="utf-8")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(processor.system, "run", fake_run)

    assert processor.process_text_files_streaming(channel) is False
    assert not any(file.suffix == ".tmp" for file in channel.working_dir.iterdir())
    assert not (channel.working_dir / f"{channel.name}-{processor.date_suffix}-combo.csv").exists()


def test_process_stealer_logs_moves_only_non_empty_results(
    processor: TelegramProcessor,
    channel: ChannelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (channel.working_dir / "loot.zip").write_text("zip", encoding="utf-8")
    (channel.working_dir / "credentials.csv").write_text("user,pass\n", encoding="utf-8")
    (channel.working_dir / "autofills.csv").write_text("", encoding="utf-8")

    def fake_run(cmd, check, capture_output, text):  # noqa: ANN001
        assert cmd[:2] == ["python3", processor.settings.get("stealer_log_processor", "path")]
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(processor.system, "run", fake_run)

    assert processor.process_stealer_logs(channel) is True
    assert (processor.output_dir / f"{channel.name}-{processor.date_suffix}-credentials.csv").exists()
    assert not (channel.working_dir / "autofills.csv").exists()


def test_process_channel_files_runs_text_processing_even_without_archives(
    processor: TelegramProcessor,
    channel: ChannelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(processor, "extract_archives", lambda current_channel: True)
    monkeypatch.setattr(processor, "has_archives", lambda current_dir: False)
    monkeypatch.setattr(processor, "deduplicate", lambda current_dir: None)
    monkeypatch.setattr(processor, "process_stealer_logs", lambda current_channel: False)
    monkeypatch.setattr(processor, "process_text_files_streaming", lambda current_channel: True)

    processed_channel, success = processor.process_channel_files(channel)

    assert processed_channel is channel
    assert success is True


def test_extract_archives_tries_passwords_then_falls_back(
    processor: TelegramProcessor,
    channel: ChannelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = channel.working_dir / "loot.zip"
    archive_path.write_text("zip", encoding="utf-8")

    attempts: list[str | None] = []

    def fake_extract_single_archive(current_archive: Path, password: str | None = None) -> bool:
        attempts.append(password)
        return password is None

    monkeypatch.setattr(processor, "extract_single_archive", fake_extract_single_archive)
    monkeypatch.setitem(processor.settings.settings["archive"], "max_parallel_extractions", 1)

    assert processor.extract_archives(channel) is True
    assert attempts == ["secret", None]


def test_download_channel_builds_expected_commands(
    processor: TelegramProcessor,
    channel: ChannelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    channel_dir = processor.downloads_dir / channel.name
    channel_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(processor.settings.settings["tdl"], "bandwidth_limit", 128)
    monkeypatch.setitem(processor.settings.settings["tdl"], "excluded_extensions", ["zip", "Txt"])
    monkeypatch.setattr(processor, "parse_export_file", lambda current_channel: ["one.zip"])

    def fake_run(cmd, check, capture_output, text, errors):  # noqa: ANN001
        commands.append(cmd)
        export_file = channel_dir / processor.EXPORT_FILE
        export_file.write_text(json.dumps({"messages": [{"type": "message", "file": "one.zip"}]}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stderr="")

    class FakeStdout:
        def __init__(self) -> None:
            self._lines = iter(["progress line\n", ""])

        def readline(self) -> str:
            return next(self._lines)

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, bufsize, universal_newlines):  # noqa: ANN001
            commands.append(cmd)
            self.stdout = FakeStdout()
            (channel_dir / "downloaded.zip").write_text("data", encoding="utf-8")

        def poll(self) -> int | None:
            return 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(processor.system, "run", fake_run)
    monkeypatch.setattr(processor.system, "popen", FakePopen)

    assert processor._download_channel_impl(channel) is True
    assert commands[0][:3] == ["tdl", "chat", "export"]
    assert commands[1][:2] == ["tdl", "dl"]
    assert "--limit" in commands[1]
    assert "-e" in commands[1]
    excluded_value = commands[1][commands[1].index("-e") + 1]
    assert set(excluded_value.split(",")) == {"zip", "ZIP", "txt", "TXT"}


def test_build_export_command_uses_channel_and_date_range(processor: TelegramProcessor, channel: ChannelConfig) -> None:
    export_file = processor.downloads_dir / channel.name / processor.EXPORT_FILE

    command = processor.build_export_command(channel, export_file)

    assert command == [
        "tdl",
        "chat",
        "export",
        "-c",
        "@chan",
        "-i",
        f"{processor.start_date},{processor.end_date}",
        "-o",
        str(export_file),
        "-t",
        str(processor.settings.get("tdl", "export_channel_threads", default=4)),
    ]


def test_build_download_command_applies_optional_settings(processor: TelegramProcessor, channel: ChannelConfig) -> None:
    export_file = processor.downloads_dir / channel.name / processor.EXPORT_FILE
    channel_dir = processor.downloads_dir / channel.name

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(processor.settings.settings["tdl"], "bandwidth_limit", 256)
    monkeypatch.setitem(processor.settings.settings["tdl"], "excluded_extensions", ["zip", "Txt"])

    try:
        command = processor.build_download_command(export_file, channel_dir)
    finally:
        monkeypatch.undo()

    assert command[:2] == ["tdl", "dl"]
    assert "--limit" in command
    assert command[command.index("--limit") + 1] == "256"
    assert "-e" in command
    assert set(command[command.index("-e") + 1].split(",")) == {"zip", "ZIP", "txt", "TXT"}


def test_list_downloaded_files_ignores_tmp_and_export_file(processor: TelegramProcessor, channel: ChannelConfig) -> None:
    (channel.working_dir / "keep.zip").write_text("data", encoding="utf-8")
    (channel.working_dir / "skip.tmp").write_text("data", encoding="utf-8")
    (channel.working_dir / processor.EXPORT_FILE).write_text("{}", encoding="utf-8")

    files = processor.list_downloaded_files(channel.working_dir)

    assert [file.name for file in files] == ["keep.zip"]


def test_get_non_empty_text_files_filters_by_min_size(processor: TelegramProcessor, channel: ChannelConfig) -> None:
    (channel.working_dir / "empty.txt").write_text("", encoding="utf-8")
    (channel.working_dir / "keep.txt").write_text("abc\n", encoding="utf-8")
    (channel.working_dir / "ignore.csv").write_text("abc\n", encoding="utf-8")
    processor.settings.settings["processing"]["min_file_size_bytes"] = 1

    files = processor.get_non_empty_text_files(channel)

    assert [file.name for file in files] == ["keep.txt"]


def test_build_sort_command_uses_settings_and_temp_file(processor: TelegramProcessor, channel: ChannelConfig) -> None:
    temp_file = channel.working_dir / "combined.txt"
    temp_file.write_text("x\n", encoding="utf-8")
    processor.settings.settings["sort"]["temp_dir"] = "/custom-tmp"
    processor.settings.settings["sort"]["memory_percent"] = 55
    processor.settings.settings["sort"]["max_parallel"] = 8

    command = processor.build_sort_command(channel, temp_file)

    assert command == [
        "sort",
        "-T",
        "/custom-tmp",
        "-u",
        "-S",
        "55%",
        "--parallel",
        "8",
        "-o",
        f"{channel.name}-{processor.date_suffix}-combo.csv",
        str(temp_file),
    ]


def test_stream_download_progress_tracks_new_files(
    processor: TelegramProcessor,
    channel: ChannelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel_dir = channel.working_dir
    expected_files = ["one.zip", "two.zip"]
    progress_messages: list[str] = []
    printed: list[str] = []

    class FakeStdout:
        def __init__(self) -> None:
            self._lines = iter(["progress 1\n", "progress 2\n", ""])

        def readline(self) -> str:
            value = next(self._lines)
            if value == "progress 1\n":
                (channel_dir / "one.zip").write_text("1", encoding="utf-8")
            if value == "progress 2\n":
                (channel_dir / "two.zip").write_text("2", encoding="utf-8")
            return value

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FakeStdout()

        def poll(self) -> int | None:
            return 0

    monkeypatch.setattr(download_module.logger, "info", lambda message, *args: progress_messages.append(message % args))
    monkeypatch.setattr("builtins.print", lambda value: printed.append(value))

    processor.stream_download_progress(
        FakeProcess(),
        channel,
        channel_dir,
        initial_files=set(),
        expected_files=expected_files,
    )

    assert printed == ["progress 1", "progress 2"]
    assert any("Downloaded 1/2 files for channel chan" in message for message in progress_messages)
    assert any("Downloaded 2/2 files for channel chan" in message for message in progress_messages)


def test_summarize_extraction_results_reports_partial_success(
    processor: TelegramProcessor,
    channel: ChannelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []

    monkeypatch.setattr(download_module.logger, "warning", lambda message, *args: warnings.append(message % args))

    result = processor.summarize_extraction_results(
        channel,
        total_count=3,
        success_count=2,
        failed_archives=["one.zip", "two.zip"],
    )

    assert result is True
    assert warnings == [
        "Extracted 2 out of 3 archives for channel chan (failed: one.zip, two.zip)"
    ]


def test_build_extract_command_includes_patterns_and_password(
    processor: TelegramProcessor,
    channel: ChannelConfig,
) -> None:
    archive_path = channel.working_dir / "loot.zip"
    archive_path.write_text("zip", encoding="utf-8")

    command = processor.build_extract_command(archive_path, "secret")

    assert command == [
        "7z",
        "x",
        "loot.zip",
        "-aoa",
        "-y",
        "-bd",
        "-ir!*.txt",
        "-ir!*.csv",
        "-ir!*pass*",
        "-ir!*auto*",
        "-psecret",
    ]


def test_is_extract_failure_output_detects_known_7z_errors(processor: TelegramProcessor) -> None:
    assert processor.is_extract_failure_output("Wrong password") is True
    assert processor.is_extract_failure_output("ERROR: Cannot convert file") is True
    assert processor.is_extract_failure_output("Everything is Ok") is False


def test_promote_result_file_moves_non_empty_output(processor: TelegramProcessor, channel: ChannelConfig) -> None:
    source = channel.working_dir / "credentials.csv"
    source.write_text("user,pass\n", encoding="utf-8")

    moved = {}

    def fake_move(src: Path, dst: Path) -> None:
        moved["src"] = src
        moved["dst"] = dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        src.unlink()

    processor.safe_move = fake_move  # type: ignore[method-assign]

    result = processor.promote_result_file(channel, "credentials.csv")

    assert result is True
    assert moved["dst"] == processor.output_dir / f"{channel.name}-{processor.date_suffix}-credentials.csv"


def test_promote_result_file_deletes_empty_output(processor: TelegramProcessor, channel: ChannelConfig) -> None:
    source = channel.working_dir / "autofills.csv"
    source.write_text("", encoding="utf-8")

    result = processor.promote_result_file(channel, "autofills.csv")

    assert result is False
    assert not source.exists()


def test_combine_text_files_writes_combined_content_and_skips_bad_file(
    processor: TelegramProcessor,
    channel: ChannelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = channel.working_dir / "first.txt"
    second = channel.working_dir / "second.txt"
    bad = channel.working_dir / "bad.txt"
    first.write_text("a\n", encoding="utf-8")
    second.write_text("b\n", encoding="utf-8")
    bad.write_text("ignore\n", encoding="utf-8")

    original_open = open

    def fake_open(path, *args, **kwargs):  # noqa: ANN001
        if Path(path) == bad:
            raise OSError("boom")
        return original_open(path, *args, **kwargs)

    warnings: list[str] = []
    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(download_module.logger, "warning", lambda message, *args: warnings.append(message % args))

    combined_path = processor.combine_text_files(channel, [first, bad, second])

    assert combined_path is not None
    assert combined_path.read_text(encoding="utf-8") == "a\nb\n"
    assert warnings == [f"Error reading {bad}: boom"]


def test_try_extract_with_passwords_returns_password_used_when_successful(
    processor: TelegramProcessor,
    channel: ChannelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = channel.working_dir / "loot.zip"
    archive_path.write_text("zip", encoding="utf-8")
    channel.passwords = ["bad", "good"]

    def fake_extract(current_archive: Path, password: str | None = None) -> bool:
        return password == "good"

    monkeypatch.setattr(processor, "extract_single_archive", fake_extract)

    result = processor.try_extract_with_passwords(channel, archive_path)

    assert result == (archive_path, True, "good")


class FakeSystem:
    def __init__(self) -> None:
        self.which_calls: list[str] = []
        self.run_calls: list[list[str]] = []
        self.popen_calls: list[list[str]] = []

    def which(self, tool: str) -> str | None:
        self.which_calls.append(tool)
        return None if tool == "tdl" else f"/usr/bin/{tool}"

    def run(self, cmd, **kwargs):  # noqa: ANN001
        self.run_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stderr="")

    def popen(self, cmd, **kwargs):  # noqa: ANN001
        self.popen_calls.append(cmd)
        raise AssertionError("popen should not be called in this test")


def test_check_dependencies_uses_injected_system_ops(tmp_path: Path) -> None:
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
        system_ops=fake_system,
    )

    processor.check_dependencies()

    assert fake_system.which_calls == ["7z", "rdfind", "sort"]
