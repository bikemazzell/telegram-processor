from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import pytest

import telegram_processor
from telegram_processor_lib import ChannelCheckResult, ChannelChecker


def test_load_rows_skips_commented_and_blank_lines(tmp_path: Path) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text(
        "\n".join(
            [
                "name,channel,password",
                "# old,@old,secret",
                "",
                "alpha,@alpha,one",
                "  # beta,@beta,two",
                "gamma,@gamma,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    checker = ChannelChecker(input_file=input_file)

    rows = checker.load_rows()

    assert [row.name for row in rows] == ["alpha", "gamma"]
    assert [row.channel for row in rows] == ["@alpha", "@gamma"]
    assert [row.line_number for row in rows] == [4, 6]


def test_comment_out_inactive_rows_preserves_existing_comments_and_adds_reason(tmp_path: Path) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text(
        "\n".join(
            [
                "name,channel,password",
                "alpha,@alpha,one",
                "# already,@gone,secret",
                "beta,@beta,two",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    checker = ChannelChecker(input_file=input_file)
    results = [
        ChannelCheckResult(name="alpha", channel="@alpha", status="active"),
        ChannelCheckResult(name="beta", channel="@beta", status="inaccessible"),
    ]

    checker.comment_out_inactive(results)

    assert input_file.read_text(encoding="utf-8").splitlines() == [
        "name,channel,password",
        "alpha,@alpha,one",
        "# already,@gone,secret",
        "# beta,@beta,two # inaccessible",
    ]


@pytest.mark.parametrize(
    ("completed_process", "expected_status"),
    [
        (subprocess.CompletedProcess(args=["tdl"], returncode=0, stdout="", stderr=""), "active"),
        (
            subprocess.CompletedProcess(
                args=["tdl"],
                returncode=1,
                stdout="",
                stderr="rpc error: CHANNEL_PRIVATE",
            ),
            "inaccessible",
        ),
        (
            subprocess.CompletedProcess(
                args=["tdl"],
                returncode=1,
                stdout="",
                stderr="rpc error: USERNAME_NOT_OCCUPIED",
            ),
            "not_found",
        ),
        (
            subprocess.CompletedProcess(
                args=["tdl"],
                returncode=1,
                stdout="",
                stderr="network timeout",
            ),
            "error",
        ),
    ],
)
def test_classify_result_maps_tdl_output_to_channel_status(
    tmp_path: Path,
    completed_process: subprocess.CompletedProcess[str],
    expected_status: str,
) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nalpha,@alpha\n", encoding="utf-8")
    checker = ChannelChecker(input_file=input_file)

    assert checker.classify_result(completed_process) == expected_status


def test_main_check_channels_runs_checker_without_dates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "channels.csv"
    input_file.write_text("name,channel\nalpha,@alpha\n", encoding="utf-8")
    calls: list[tuple[str, bool]] = []

    parsed_args = argparse.Namespace(
        input=str(input_file),
        start=None,
        end=None,
        output_dir=str(tmp_path / "output"),
        download_dir=str(tmp_path / "downloads"),
        settings=None,
        verbose=False,
        process_only=False,
        auto_clean=False,
        check_channels=True,
        comment_missing=True,
    )

    monkeypatch.setattr(telegram_processor.argparse.ArgumentParser, "parse_args", lambda self: parsed_args)
    monkeypatch.setattr(telegram_processor, "setup_logging", lambda verbose, settings=None: logging.getLogger("test"))
    monkeypatch.setattr(
        telegram_processor.ChannelChecker,
        "run",
        lambda self, comment_missing=False: calls.append((str(self.input_file), comment_missing)),
    )

    telegram_processor.main()

    assert calls == [(str(input_file), True)]
