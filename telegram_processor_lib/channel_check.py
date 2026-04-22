from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess

from .config import ProcessingError
from .logging_utils import logger
from .system_ops import SystemOps


@dataclass(frozen=True)
class ChannelRow:
    """One active CSV row to validate."""

    line_number: int
    raw_line: str
    name: str
    channel: str


@dataclass(frozen=True)
class ChannelCheckResult:
    """Validation result for a configured channel."""

    name: str
    channel: str
    status: str
    line_number: int = 0

    @property
    def is_active(self) -> bool:
        return self.status == "active"


class ChannelChecker:
    """Validate configured channels and optionally comment out inactive rows."""

    def __init__(self, input_file: str | Path, system_ops: SystemOps | None = None) -> None:
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise ProcessingError(f"Input file not found: {input_file}")
        self.system = system_ops or SystemOps()

    def check_dependencies(self) -> None:
        if not self.system.which("tdl"):
            raise ProcessingError("Missing required tool:\ntdl: Telegram downloader (tdl) - Install from: https://github.com/iyear/tdl")

    def load_rows(self) -> list[ChannelRow]:
        lines = self.input_file.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ProcessingError("CSV file is empty")

        header_line = lines[0]
        header = next(csv.reader([header_line]))
        if len(header) < 2 or header[0].strip().lower() != "name" or header[1].strip().lower() != "channel":
            raise ProcessingError("CSV must start with 'name' and 'channel' columns")

        rows: list[ChannelRow] = []
        for line_number, raw_line in enumerate(lines[1:], start=2):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            values = next(csv.reader([raw_line]))
            if len(values) < 2 or not values[0].strip() or not values[1].strip():
                continue

            rows.append(
                ChannelRow(
                    line_number=line_number,
                    raw_line=raw_line,
                    name=values[0].strip(),
                    channel=values[1].strip(),
                )
            )

        return rows

    def build_check_command(self, channel: str, output_file: Path) -> list[str]:
        return [
            "tdl",
            "chat",
            "export",
            "-c",
            channel,
            "-T",
            "last",
            "-i",
            "1",
            "-o",
            str(output_file),
        ]

    def classify_result(self, completed_process: CompletedProcess[str]) -> str:
        if completed_process.returncode == 0:
            return "active"

        output = f"{completed_process.stdout}\n{completed_process.stderr}".lower()
        if any(marker in output for marker in ("username_not_occupied", "chat not found", "not found", "no user has")):
            return "not_found"
        if any(marker in output for marker in ("channel_private", "private", "forbidden", "invite request needed")):
            return "inaccessible"
        return "error"

    def check_row(self, row: ChannelRow) -> ChannelCheckResult:
        with tempfile.NamedTemporaryFile(prefix="tdl-check-", suffix=".json", delete=False) as handle:
            output_file = Path(handle.name)

        try:
            result = self.system.run(
                self.build_check_command(row.channel, output_file),
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
            )
        finally:
            output_file.unlink(missing_ok=True)

        status = self.classify_result(result)
        return ChannelCheckResult(
            name=row.name,
            channel=row.channel,
            status=status,
            line_number=row.line_number,
        )

    def comment_out_inactive(self, results: list[ChannelCheckResult]) -> None:
        lines = self.input_file.read_text(encoding="utf-8").splitlines()
        rows_by_identity = {
            (row.line_number, row.name, row.channel): row
            for row in self.load_rows()
        }

        for result in results:
            if result.is_active:
                continue

            target_line_number = result.line_number
            if target_line_number <= 0:
                for row in rows_by_identity.values():
                    if row.name == result.name and row.channel == result.channel:
                        target_line_number = row.line_number
                        break

            if target_line_number <= 0 or target_line_number > len(lines):
                continue

            original_line = lines[target_line_number - 1]
            if original_line.lstrip().startswith("#"):
                continue

            lines[target_line_number - 1] = f"# {original_line} # {result.status}"

        self.input_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self, comment_missing: bool = False) -> list[ChannelCheckResult]:
        self.check_dependencies()
        rows = self.load_rows()
        results = [self.check_row(row) for row in rows]

        for result in results:
            logger.info("Channel %s (%s): %s", result.name, result.channel, result.status)

        if comment_missing:
            self.comment_out_inactive(results)

        return results
