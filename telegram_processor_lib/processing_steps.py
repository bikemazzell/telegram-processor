from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .logging_utils import logger


class ProcessingStepsMixin:
    """Per-channel processing steps after download/extraction."""

    def get_non_empty_text_files(self, channel) -> list[Path]:
        min_file_size = self.settings.get("processing", "min_file_size_bytes", default=0)
        return [
            txt_file
            for txt_file in channel.working_dir.glob("*.txt")
            if txt_file.stat().st_size > min_file_size
        ]

    def build_sort_command(self, channel, temp_combined_path: Path) -> list[str]:
        sort_settings = self.settings.get("sort", default={})
        output_file = f"{channel.name}-{self.date_suffix}-combo.csv"
        return [
            "sort",
            "-T",
            str(sort_settings.get("temp_dir", "/tmp")),
            "-u",
            "-S",
            f"{sort_settings.get('memory_percent', 30)}%",
            "--parallel",
            str(sort_settings.get("max_parallel", 16)),
            "-o",
            str(output_file),
            str(temp_combined_path),
        ]

    def promote_result_file(self, channel, filename: str) -> bool:
        source = channel.working_dir / filename
        if not source.exists():
            return False

        min_result_size = self.settings.get("processing", "min_result_file_size_bytes", default=1)
        if source.stat().st_size > min_result_size:
            new_name = f"{channel.name}-{self.date_suffix}-{filename}"
            self.safe_move(source, self.output_dir / new_name)
            return True

        source.unlink()
        return False

    def combine_text_files(self, channel, text_files: list[Path]) -> Path | None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=channel.working_dir, suffix=".txt") as temp_combined:
            temp_combined_path = Path(temp_combined.name)
            for txt_file in text_files:
                try:
                    with open(txt_file, "r", encoding="utf-8", errors="replace") as infile:
                        for line in infile:
                            temp_combined.write(line)
                except Exception as exc:
                    logger.warning("Error reading %s: %s", txt_file, exc)
                    continue

        if temp_combined_path.stat().st_size == 0:
            temp_combined_path.unlink(missing_ok=True)
            return None

        return temp_combined_path

    def deduplicate(self, directory) -> None:
        if not directory or not directory.exists():
            return

        try:
            self.system.run(
                ["rdfind", "-deleteduplicates", "true", "."],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Deduplication completed for %s", directory)
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to deduplicate files: %s", exc.stderr)
        finally:
            results_file = directory / self.DEDUPLICATION_FILE
            if results_file.exists():
                results_file.unlink()

    def process_text_files_streaming(self, channel) -> bool:
        if not channel.working_dir or not channel.working_dir.exists():
            return False

        output_file = f"{channel.name}-{self.date_suffix}-combo.csv"

        try:
            txt_files = list(channel.working_dir.glob("*.txt"))
            if not txt_files:
                logger.info("No text files found in root directory of %s", channel.name)
                return False

            non_empty_files = self.get_non_empty_text_files(channel)
            if not non_empty_files:
                logger.info("No non-empty text files found in root directory of %s", channel.name)
                return False

            temp_combined_path = self.combine_text_files(channel, non_empty_files)
            if temp_combined_path is None:
                logger.info("No valid content found in text files for %s", channel.name)
                return False

            sort_cmd = self.build_sort_command(channel, temp_combined_path)
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            self.system.run(sort_cmd, check=True, cwd=channel.working_dir, env=env)

            temp_combined_path.unlink(missing_ok=True)
            min_result_size = self.settings.get("processing", "min_result_file_size_bytes", default=1)
            output_path = channel.working_dir / output_file
            if not output_path.exists() or output_path.stat().st_size <= min_result_size:
                logger.info("No unique content found in %s", channel.name)
                output_path.unlink(missing_ok=True)
                return False

            self.safe_move(output_path, self.output_dir / output_file)
            return True
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.error("Failed to process text files for %s: %s", channel.name, str(exc))
            if "temp_combined_path" in locals():
                temp_combined_path.unlink(missing_ok=True)
            (channel.working_dir / output_file).unlink(missing_ok=True)
            return False

    def process_stealer_logs(self, channel) -> bool:
        if not channel.working_dir or not channel.working_dir.exists():
            return False
        if not self.has_archives(channel.working_dir):
            return False

        try:
            processor_path = self.settings.get("stealer_log_processor", "path", default=None)
            if not processor_path:
                logger.error("Stealer log processor path not configured")
                return False

            self.system.run(
                ["python3", processor_path, str(channel.working_dir)],
                check=True,
                capture_output=True,
                text=True,
            )

            has_results = False
            for filename in ["credentials.csv", "autofills.csv"]:
                if self.promote_result_file(channel, filename):
                    has_results = True
            return has_results
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to process stealer logs for %s: %s", channel.name, exc.stderr)
            return False
        except OSError as exc:
            logger.error("Failed to move result files for %s: %s", channel.name, str(exc))
            return False

    def process_channel_files(self, channel):
        try:
            logger.info("Processing downloaded files for channel: %s", channel.name)
            extraction_success = self.extract_archives(channel)
            if extraction_success and channel.working_dir and channel.working_dir.exists():
                logger.info("Deduplicating extracted files for %s", channel.name)
                self.deduplicate(channel.working_dir)

            processing_success = False
            if extraction_success and self.has_archives(channel.working_dir):
                stealer_success = self.process_stealer_logs(channel)
                if stealer_success:
                    processing_success = True
                    logger.info("Successfully processed stealer logs for %s", channel.name)

            text_success = self.process_text_files_streaming(channel)
            if text_success:
                processing_success = True
                logger.info("Successfully processed text files for %s", channel.name)

            if processing_success:
                return channel, True

            logger.warning("No valid results found for %s", channel.name)
            return channel, False
        except Exception as exc:
            logger.error("Error processing %s: %s", channel.name, str(exc))
            return channel, False
