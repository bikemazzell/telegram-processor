from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import as_completed
from pathlib import Path
from typing import Optional, Tuple

from .config import ChannelConfig
from .logging_utils import logger


class RetryMixin:
    """Retry helper for transient external command failures."""

    def retry_operation(self, func, *args, max_retries: int = 3, retry_delay: int = 5, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                wait_time = retry_delay * (2**attempt)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %d seconds...",
                    attempt + 1,
                    max_retries,
                    str(exc),
                    wait_time,
                )
                time.sleep(wait_time)


class DownloadMixin(RetryMixin):
    """Channel download and archive extraction behavior."""

    def build_export_command(self, channel: ChannelConfig, export_file: Path) -> list[str]:
        return [
            "tdl",
            "chat",
            "export",
            "-c",
            channel.channel,
            "-i",
            f"{self.start_date},{self.end_date}",
            "-o",
            str(export_file),
            "-t",
            str(self.settings.get("tdl", "export_channel_threads", default=4)),
        ]

    def build_download_command(self, export_file: Path, channel_dir: Path) -> list[str]:
        dl_cmd = [
            "tdl",
            "dl",
            "-l",
            str(self.settings.get("tdl", "max_parallel_downloads", default=4)),
            "-f",
            str(export_file),
            "--reconnect-timeout",
            str(self.settings.get("tdl", "reconnect_timeout", default=0)),
            "--skip-same",
            "-d",
            str(channel_dir),
            "--continue",
        ]

        bandwidth_limit = self.settings.get("tdl", "bandwidth_limit", default=0)
        if bandwidth_limit > 0:
            dl_cmd.extend(["--limit", str(bandwidth_limit)])

        excluded_extensions = self.settings.get("tdl", "excluded_extensions", default=[])
        if excluded_extensions:
            excluded_extensions = self.settings.prepare_extensions(excluded_extensions)
            dl_cmd.extend(["-e", ",".join(excluded_extensions)])

        return dl_cmd

    def list_downloaded_files(self, channel_dir: Path) -> list[Path]:
        return [
            file
            for file in channel_dir.iterdir()
            if file.is_file() and not file.name.endswith(".tmp") and file.name != self.EXPORT_FILE
        ]

    def stream_download_progress(
        self,
        process,
        channel: ChannelConfig,
        channel_dir: Path,
        initial_files: set[str],
        expected_files: list[str],
    ) -> None:
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())
                if "downloading" in output.lower() or "progress" in output.lower():
                    current_files = {file.name for file in self.list_downloaded_files(channel_dir)}
                    new_files = current_files - initial_files
                    if new_files:
                        logger.info("Downloaded %d/%d files for channel %s", len(new_files), len(expected_files), channel.name)

    def summarize_extraction_results(
        self,
        channel: ChannelConfig,
        total_count: int,
        success_count: int,
        failed_archives: list[str],
    ) -> bool:
        if success_count == 0 and total_count > 0:
            logger.error("Failed to extract any archives for channel %s", channel.name)
            return False

        if success_count < total_count:
            logger.warning(
                "Extracted %d out of %d archives for channel %s (failed: %s)",
                success_count,
                total_count,
                channel.name,
                ", ".join(failed_archives[:5]) + ("..." if len(failed_archives) > 5 else ""),
            )
        else:
            logger.info("Successfully extracted all %d archives for channel %s", total_count, channel.name)

        return True

    def build_extract_command(self, archive_path: Path, password: Optional[str] = None) -> list[str]:
        patterns = self.settings.get("archive", "extract_patterns", default=["*.txt", "*.csv"])
        command = ["7z", "x", str(archive_path.name), "-aoa", "-y", "-bd"]
        for pattern in patterns:
            command.extend(["-ir!" + pattern])
        if password:
            command.extend(["-p" + password])
        return command

    def is_extract_failure_output(self, stderr: str) -> bool:
        return "ERROR: Cannot convert" in stderr or "Wrong password" in stderr

    def try_extract_with_passwords(
        self,
        channel: ChannelConfig,
        archive_path: Path,
    ) -> Tuple[Path, bool, Optional[str]]:
        if self.verbose:
            logger.info("Processing archive %s", archive_path.name)

        if channel.has_passwords:
            for password in channel.passwords:
                if self.extract_single_archive(archive_path, password):
                    return (archive_path, True, password)

        if self.extract_single_archive(archive_path):
            return (archive_path, True, None)

        return (archive_path, False, None)

    def download_channel_with_retry(self, channel: ChannelConfig) -> bool:
        max_retries = self.settings.get("tdl", "max_retries", default=3)
        retry_delay = self.settings.get("tdl", "retry_delay", default=5)
        return self.retry_operation(
            self._download_channel_impl,
            channel,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def _download_channel_impl(self, channel: ChannelConfig) -> bool:
        logger.info("Downloading channel: %s", channel.name)
        channel_dir = self.downloads_dir / channel.name
        export_file = channel_dir / self.EXPORT_FILE

        try:
            channel_dir.mkdir(parents=True, exist_ok=True)
            export_cmd = self.build_export_command(channel, export_file)

            if self.verbose:
                logger.info("Running command: %s", " ".join(export_cmd))

            export_result = self.system.run(export_cmd, check=True, capture_output=True, text=True, errors="replace")
            if self.verbose and export_result.stderr:
                logger.info("tdl export output for %s:\n%s", channel.name, export_result.stderr.strip())

            expected_files = self.parse_export_file(channel)
            if not expected_files:
                logger.info("No files found to download for channel %s in the specified date range", channel.name)
                return False

            logger.info("Found %d files to download for channel %s", len(expected_files), channel.name)

            dl_cmd = self.build_download_command(export_file, channel_dir)

            if self.verbose:
                logger.info("Running command: %s", " ".join(dl_cmd))

            logger.info("Starting download of %d files for channel %s...", len(expected_files), channel.name)
            initial_files = {file.name for file in self.list_downloaded_files(channel_dir)}

            try:
                process = self.system.popen(
                    dl_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                self.stream_download_progress(process, channel, channel_dir, initial_files, expected_files)

                return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, dl_cmd)
            except subprocess.CalledProcessError as exc:
                logger.error("Download failed for %s with exit code %d", channel.name, exc.returncode)
                raise

            downloaded_files = self.list_downloaded_files(channel_dir)
            if not downloaded_files:
                logger.warning("Download command completed but no files were found for channel %s", channel.name)
                return False

            logger.info("Successfully downloaded %d files for channel %s", len(downloaded_files), channel.name)
            return True
        except subprocess.CalledProcessError as exc:
            error_msg = f"Failed to process channel {channel.name}"
            if hasattr(exc, "stderr") and exc.stderr:
                error_msg += f": {exc.stderr}"
            logger.error(error_msg)
            raise
        except Exception as exc:
            logger.error("Unexpected error downloading channel %s: %s", channel.name, str(exc))
            raise
        finally:
            root_export_file = Path(self.EXPORT_FILE)
            if root_export_file.exists():
                root_export_file.unlink()

    def download_channel_wrapper(self, channel: ChannelConfig) -> Tuple[ChannelConfig, bool, Optional[str]]:
        try:
            if self.processed_channels.get(channel.name, False):
                logger.info("Channel %s already processed, skipping", channel.name)
                channel.working_dir = self.downloads_dir / channel.name
                return channel, True, None

            success = self.download_channel_with_retry(channel)
            if success:
                channel.working_dir = self.downloads_dir / channel.name
                if not channel.working_dir.exists() or not any(channel.working_dir.iterdir()):
                    return channel, False, f"No files found in {channel.working_dir}"
                self.processed_channels[channel.name] = True
                self.save_checkpoint()
                return channel, True, None
            return channel, False, "No files downloaded"
        except Exception as exc:
            return channel, False, str(exc)

    def get_archive_files(self, directory: Path):
        if not directory.exists():
            return []

        supported_extensions = self.settings.get("archive", "supported_extensions", default=[".zip", ".rar", ".7z"])
        archive_extensions = {ext.lower() for ext in supported_extensions}
        return [entry for entry in directory.iterdir() if entry.is_file() and entry.suffix.lower() in archive_extensions]

    def extract_single_archive(self, archive_path: Path, password: Optional[str] = None) -> bool:
        max_retries = self.settings.get("archive", "max_retries", default=2)
        for attempt in range(max_retries):
            try:
                if self._extract_archive_impl(archive_path, password):
                    return True
                if attempt < max_retries - 1 and self.verbose:
                    logger.warning("Extraction failed for %s, retrying (%d/%d)...", archive_path.name, attempt + 1, max_retries)
            except Exception as exc:
                if attempt == max_retries - 1:
                    logger.error("Failed to extract %s after %d attempts: %s", archive_path.name, max_retries, str(exc))
                    return False
        return False

    def _extract_archive_impl(self, archive_path: Path, password: Optional[str] = None) -> bool:
        try:
            base_cmd = self.build_extract_command(archive_path, password)

            if self.verbose:
                logger.info("Running command: %s", " ".join(base_cmd))
                logger.info("Working directory: %s", archive_path.parent)

            timeout = self.settings.get("archive", "extract_timeout", default=300)
            env = os.environ.copy()
            env["LANG"] = "C.UTF-8"

            with self.managed_process(
                base_cmd,
                cwd=archive_path.parent,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                encoding="utf-8",
                errors="replace",
            ) as process:
                try:
                    _, stderr = process.communicate(timeout=timeout)
                    if stderr and self.is_extract_failure_output(stderr):
                        if self.verbose:
                            logger.error("7z extraction failed for %s:\n%s", archive_path.name, stderr[:500])
                        return False
                    if process.returncode != 0:
                        if self.verbose:
                            logger.error("7z extraction failed for %s with exit code %d", archive_path.name, process.returncode)
                        return False
                    return True
                except subprocess.TimeoutExpired:
                    logger.error("Extraction timed out after %d seconds for %s", timeout, archive_path.name)
                    return False
        except Exception as exc:
            logger.error("Failed to run extraction for %s: %s", archive_path.name, str(exc))
            return False

    def extract_archives(self, channel: ChannelConfig) -> bool:
        if not channel.working_dir or not channel.working_dir.exists():
            return False

        archive_files = self.get_archive_files(channel.working_dir)
        if not archive_files:
            return True

        total_count = len(archive_files)
        max_workers = self.settings.get("archive", "max_parallel_extractions", default=4)
        actual_workers = min(max_workers, total_count)

        if actual_workers > 1:
            logger.info(
                "Starting parallel extraction with %d workers for %d archives in channel %s",
                actual_workers,
                total_count,
                channel.name,
            )

        success_count = 0
        failed_archives = []

        with self.extraction_executor_cls(max_workers=actual_workers) as executor:
            futures = {executor.submit(self.try_extract_with_passwords, channel, archive): archive for archive in archive_files}
            for future in as_completed(futures):
                archive_path, success, password_used = future.result()
                if success:
                    success_count += 1
                    if self.verbose:
                        if password_used:
                            logger.info("Successfully extracted %s with password", archive_path.name)
                        else:
                            logger.info("Successfully extracted %s without password", archive_path.name)
                else:
                    failed_archives.append(archive_path.name)
                    if self.verbose:
                        logger.warning("Failed to extract %s with any method", archive_path.name)

        return self.summarize_extraction_results(channel, total_count, success_count, failed_archives)

    def has_archives(self, directory: Path) -> bool:
        return bool(self.get_archive_files(directory))
