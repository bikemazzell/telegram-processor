from __future__ import annotations

import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import ChannelConfig, ProcessingError, Settings
from .logging_utils import logger
from .system_ops import SystemOps


class BaseProcessorMixin:
    """Core configuration, filesystem, and process helpers."""

    CHECKPOINT_FILE = ".processing_checkpoint.json"
    EXPORT_FILE = "tdl-export.json"
    DEDUPLICATION_FILE = "results.txt"

    def __init__(
        self,
        input_file: str | Path,
        start_date: str,
        end_date: str,
        output_dir: Optional[str | Path] = None,
        download_dir: Optional[str | Path] = None,
        settings_file: Optional[str | Path] = None,
        verbose: bool = False,
        process_only: bool = False,
        auto_clean: bool = False,
        system_ops: Optional[SystemOps] = None,
        process_executor_cls=ProcessPoolExecutor,
        extraction_executor_cls=ThreadPoolExecutor,
    ) -> None:
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise ProcessingError(f"Input file not found: {input_file}")

        self.start_date = start_date
        self.end_date = end_date
        self.process_only = process_only
        self.auto_clean = auto_clean

        start_datetime = datetime.fromtimestamp(float(start_date))
        self.date_suffix = start_datetime.strftime("%m-%Y")

        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "output"
        self.downloads_dir = Path(download_dir) if download_dir else Path.cwd() / "downloads"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.downloads_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise ProcessingError(f"Failed to create required directories: {exc}")

        self.channels: List[ChannelConfig] = []
        self.settings = Settings(settings_file)
        self.verbose = verbose
        self.system = system_ops or SystemOps()
        self.process_executor_cls = process_executor_cls
        self.extraction_executor_cls = extraction_executor_cls
        self.processed_channels: Dict[str, bool] = {}
        self.checkpoint_file = self.downloads_dir / self.CHECKPOINT_FILE

    def check_dependencies(self) -> None:
        required_tools = {
            "7z": "7-Zip archiver - Install: apt-get install p7zip-full",
            "rdfind": "Duplicate file finder - Install: apt-get install rdfind",
            "sort": "GNU sort utility - Install via coreutils package",
        }
        if not self.process_only:
            required_tools["tdl"] = "Telegram downloader (tdl) - Install from: https://github.com/iyear/tdl"

        missing_tools = []
        for tool, description in required_tools.items():
            if not self.system.which(tool):
                missing_tools.append(f"{tool}: {description}")

        if missing_tools:
            error_msg = "Missing required tools:\n" + "\n".join(missing_tools)
            raise ProcessingError(error_msg)

    def load_checkpoint(self) -> None:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as handle:
                    self.processed_channels = json.load(handle)
                logger.info("Loaded checkpoint with %d processed channels", len(self.processed_channels))
            except Exception as exc:
                logger.warning("Failed to load checkpoint: %s", exc)
                self.processed_channels = {}

    def save_checkpoint(self) -> None:
        try:
            with open(self.checkpoint_file, "w") as handle:
                json.dump(self.processed_channels, handle)
        except Exception as exc:
            logger.warning("Failed to save checkpoint: %s", exc)

    def load_channels(self) -> None:
        try:
            with open(self.input_file, "r", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration:
                    raise ProcessingError("CSV file is empty")

                if len(header) < 2 or header[0].strip().lower() != "name" or header[1].strip().lower() != "channel":
                    raise ProcessingError("CSV must start with 'name' and 'channel' columns")

                for row in reader:
                    if not row or not row[0] or not row[1]:
                        continue

                    channel = ChannelConfig(
                        name=row[0],
                        channel=row[1],
                        passwords=row[2:] if len(row) > 2 else [],
                    )
                    self.channels.append(channel)

            if not self.channels:
                raise ProcessingError("No channels found in CSV file")
        except csv.Error as exc:
            raise ProcessingError(f"Failed to parse CSV file: {exc}")

    def cleanup_temp_files(self) -> None:
        try:
            cleanup_patterns = self.settings.get(
                "logging",
                "temp_cleanup_patterns",
                default=["tdl-export*.json", "*.temp", "sort*"],
            )
            if cleanup_patterns is None:
                cleanup_patterns = ["tdl-export*.json", "*.temp", "sort*"]

            for pattern in cleanup_patterns:
                if pattern != "sort*":
                    for temp_file in Path().glob(pattern):
                        temp_file.unlink(missing_ok=True)

            temp_dir = Path(self.settings.get("sort", "temp_dir", default="/tmp"))
            for sort_temp in temp_dir.glob("sort*"):
                if sort_temp.is_file():
                    sort_temp.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to clean up temporary files: %s", exc)

    @contextmanager
    def managed_process(self, *args, **kwargs):
        process = self.system.popen(*args, **kwargs)
        try:
            yield process
        finally:
            if process.poll() is None:
                terminate_timeout = self.settings.get("subprocess", "terminate_timeout", default=5)
                process.terminate()
                try:
                    process.wait(timeout=terminate_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def safe_move(self, src: Path, dst: Path) -> None:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            self.system.move(str(src), str(dst))
        except Exception as exc:
            raise ProcessingError(f"Failed to move {src} to {dst}: {exc}")

    def remove_tree(self, directory: Path) -> None:
        self.system.rmtree(directory)

    def parse_export_file(self, channel: ChannelConfig) -> List[str]:
        export_file = self.downloads_dir / channel.name / self.EXPORT_FILE
        try:
            with open(export_file, "r", encoding="utf-8") as handle:
                export_data = json.load(handle)
            files = []
            for message in export_data.get("messages", []):
                if message.get("type") == "message" and "file" in message:
                    files.append(message["file"])
            return files
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to parse export file for channel %s: %s", channel.name, str(exc))
            return []
