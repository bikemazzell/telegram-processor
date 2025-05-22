#!/usr/bin/env python3

"""
Telegram Channel Processor

This script processes Telegram channels by downloading content and processing
files based on their type (archives or text files). It handles extraction,
deduplication, and organization of the processed files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Base exception for processing errors."""

    pass


class ChannelConfig:
    """Configuration for a Telegram channel."""

    def __init__(self, name: str, channel: str, password: Optional[str] = None) -> None:
        """
        Initialize channel configuration.

        Args:
            name: Display name for the channel
            channel: Channel identifier (ID or username)
            password: Optional password for archive extraction

        Raises:
            ValueError: If name or channel is empty
        """
        if not name or not name.strip():
            raise ValueError("Channel name cannot be empty")
        if not channel or not channel.strip():
            raise ValueError("Channel identifier cannot be empty")

        self.name: str = name.strip()
        self.channel: str = channel.strip()
        self.password: Optional[str] = password.strip() if password else None
        self.working_dir: Optional[Path] = None

    @property
    def has_password(self) -> bool:
        """Check if channel has a valid password."""
        return bool(self.password and self.password.strip())


class Settings:
    """Configuration settings loaded from JSON file."""

    DEFAULT_SETTINGS_FILE = "settings.json"

    def __init__(self, settings_file: Optional[str | Path] = None) -> None:
        """
        Initialize settings from JSON file.

        Args:
            settings_file: Optional path to settings file
        """
        self.settings_file = Path(settings_file or self.DEFAULT_SETTINGS_FILE)
        self.settings: Dict[str, Any] = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        """
        Load settings from JSON file.

        Returns:
            Dict containing settings

        Raises:
            ProcessingError: If settings file is invalid
        """
        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(
                "Settings file not found: %s, using defaults", self.settings_file
            )
            return self._default_settings()
        except json.JSONDecodeError as e:
            raise ProcessingError(f"Invalid JSON in settings file: {e}")
        except Exception as e:
            raise ProcessingError(f"Failed to load settings: {e}")

    def _default_settings(self) -> Dict[str, Any]:
        """Return default settings."""
        return {
            "stealer_log_processor": {"path": "./stealer-log-processor/main.py"},
            "tdl": {
                "max_parallel_downloads": 1,
                "reconnect_timeout": 0,
                "download_timeout": 7200,
                "export_channel_threads": 4,
                "bandwidth_limit": 0,  # 0 means unlimited
                "chunk_size": 128,  # in KB
                "excluded_extensions": [
                    "jpg", "gif", "png", "webp", "webm", "mp4"
                ],
                "included_extensions": [
                    "zip", "rar", "7z", "txt", "csv"
                ],
            },
            "sort": {
                "memory_percent": 30,
                "max_parallel": 16,
                "temp_dir": "/tmp",  # More portable default
            },
            "archive": {
                "extract_patterns": [
                    "*.txt",
                    "*.csv",
                    "*pass*",
                    "*auto*"
                ]
            },
        }

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a nested setting value.

        Args:
            *keys: Sequence of keys to access nested settings
            default: Default value if key doesn't exist

        Returns:
            Setting value or default
        """
        current = self.settings
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key, default)
            if current is None:
                return default
        return current

    def _prepare_extensions(self, extensions: List[str]) -> List[str]:
        """
        Prepare extensions by ensuring both lower and upper case variants are included.
        
        Args:
            extensions: List of extensions (without leading dot)
            
        Returns:
            List with both lowercase and uppercase variants
        """
        result = set()
        for ext in extensions:
            # Add lowercase version
            ext = ext.lower().strip()
            if ext:
                # Remove leading dots if present
                if ext.startswith('.'):
                    ext = ext[1:]
                result.add(ext)
                # Add uppercase version if different
                upper_ext = ext.upper()
                if upper_ext != ext:
                    result.add(upper_ext)
        
        return list(result)


class TelegramProcessor:
    """Main processor for Telegram channels."""

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
    ) -> None:
        """
        Initialize the processor.

        Args:
            input_file: Path to CSV file with channel configurations
            start_date: Start date in epoch format
            end_date: End date in epoch format
            output_dir: Optional output directory for processed files
            download_dir: Optional directory for downloaded files
            settings_file: Optional path to settings file
            verbose: Whether to show detailed output
            process_only: Whether to skip download phase and only process existing files
            auto_clean: Whether to automatically clean up without prompting

        Raises:
            ProcessingError: If input file doesn't exist or output directory can't be created
        """
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise ProcessingError(f"Input file not found: {input_file}")

        self.start_date = start_date
        self.end_date = end_date
        self.process_only = process_only
        self.auto_clean = auto_clean

        # Convert epoch start date to month-year format for file naming
        start_datetime = datetime.fromtimestamp(float(start_date))
        self.date_suffix = start_datetime.strftime("%m-%Y")

        # Setup directories
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "output"
        self.downloads_dir = Path(download_dir) if download_dir else Path.cwd() / "downloads"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.downloads_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ProcessingError(f"Failed to create required directories: {e}")

        self.channels: List[ChannelConfig] = []
        self.settings = Settings(settings_file)
        self.verbose = verbose

    def check_dependencies(self) -> None:
        """
        Check if required external tools are available.

        Raises:
            ProcessingError: If any required tool is missing
        """
        required_tools = ["tdl", "7z", "rdfind", "sort"]
        for tool in required_tools:
            try:
                subprocess.run(["which", tool], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                raise ProcessingError(f"Required tool not found: {tool}")

    def load_channels(self) -> None:
        """
        Load channel configurations from CSV file.

        Raises:
            ProcessingError: If CSV file is invalid or missing required columns
        """
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # Strip whitespace from fieldnames
                if reader.fieldnames:
                    reader.fieldnames = [field.strip() for field in reader.fieldnames]

                if not {"name", "channel"}.issubset(reader.fieldnames or []):
                    raise ProcessingError("CSV must have 'name' and 'channel' columns")

                for row in reader:
                    # Strip whitespace from values
                    cleaned_row = {
                        k.strip(): v.strip() if v else v for k, v in row.items()
                    }
                    channel = ChannelConfig(
                        name=cleaned_row["name"],
                        channel=cleaned_row["channel"],
                        password=cleaned_row.get("password", ""),
                    )
                    self.channels.append(channel)

            if not self.channels:
                raise ProcessingError("No channels found in CSV file")

        except csv.Error as e:
            raise ProcessingError(f"Failed to parse CSV file: {e}")

    def cleanup_temp_files(self) -> None:
        """Clean up any temporary files."""
        try:
            Path("tdl-export.json").unlink(missing_ok=True)
            for path in Path().glob("*.temp"):
                path.unlink()
        except Exception as e:
            logger.warning("Failed to clean up temporary files: %s", e)

    def safe_move(self, src: Path, dst: Path) -> None:
        """
        Safely move a file with proper error handling.

        Args:
            src: Source path
            dst: Destination path

        Raises:
            ProcessingError: If move operation fails
        """
        try:
            shutil.move(str(src), str(dst))
        except Exception as e:
            raise ProcessingError(f"Failed to move {src} to {dst}: {e}")

    def terminate_process(
        self, process: subprocess.Popen[str], timeout: int = 5
    ) -> None:
        """
        Safely terminate a subprocess.

        Args:
            process: Process to terminate
            timeout: Timeout in seconds for graceful termination
        """
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def parse_export_file(self, channel: ChannelConfig) -> List[str]:
        """
        Parse tdl-export.json file to get list of expected files.

        Args:
            channel: Channel configuration

        Returns:
            List of filenames to be downloaded
        """
        export_file = self.downloads_dir / channel.name / "tdl-export.json"
        try:
            with open(export_file, "r", encoding="utf-8") as f:
                export_data = json.load(f)
                
            # More direct access to the file information
            return [
                message["file"] 
                for message in export_data.get("messages", [])
                if message.get("type") == "message" and "file" in message
            ]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(
                "Failed to parse export file for channel %s: %s", channel.name, str(e)
            )
            return []
        except KeyError as e:
            logger.error(
                "Unexpected JSON structure in export file for channel %s: %s", 
                channel.name, str(e)
            )
            return []

    def setup_channel_directory(self, channel: ChannelConfig) -> Path:
        """
        Create and setup channel-specific directory.

        Args:
            channel: Channel configuration

        Returns:
            Path: Channel-specific directory path
        """
        channel_dir = self.downloads_dir / channel.name
        try:
            if channel_dir.exists():
                shutil.rmtree(channel_dir)
            channel_dir.mkdir(parents=True, exist_ok=True)
            return channel_dir
        except Exception as e:
            raise ProcessingError(
                f"Failed to setup directory for channel {channel.name}: {e}"
            )

    def download_channel(self, channel: ChannelConfig) -> bool:
        """
        Download content from a Telegram channel.

        Args:
            channel: Channel configuration

        Returns:
            bool: True if any files were downloaded, False otherwise
        """
        logger.info("Downloading channel: %s", channel.name)

        # Setup channel directory path
        channel_dir = self.downloads_dir / channel.name
        export_file = channel_dir / "tdl-export.json"

        try:
            # Ensure channel directory exists for export file
            channel_dir.mkdir(parents=True, exist_ok=True)

            # Create export command for file downloads
            export_cmd = [
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
                str(self.settings.get("tdl", "export_channel_threads", default=4))
            ]

            # Run export
            if self.verbose:
                logger.info("Running command: %s", " ".join(export_cmd))
            
            # Run without capturing output to show progress in real-time
            subprocess.run(export_cmd, check=True, text=True)

            # Parse export file to get expected files
            expected_files = self.parse_export_file(channel)
            if not expected_files:
                logger.info(
                    "No files found to download for channel %s in the specified date range",
                    channel.name,
                )
                return False

            logger.info(
                "Found %d files to download for channel %s",
                len(expected_files),
                channel.name,
            )

            # Prepare the download command with optimized parameters
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
                "--continue"
            ]
                
            # Add bandwidth limit if specified and not zero
            bandwidth_limit = self.settings.get("tdl", "bandwidth_limit", default=0)
            if bandwidth_limit > 0:
                dl_cmd.extend(["--limit", str(bandwidth_limit)])

            # Process extensions to exclude
            excluded_extensions = self.settings.get("tdl", "excluded_extensions", default=[])
            if excluded_extensions:
                # Process extensions to handle case variants
                excluded_extensions = self.settings._prepare_extensions(excluded_extensions)
                dl_cmd.extend(["-e", ",".join(excluded_extensions)])

            # Run download with proper error handling
            try:
                if self.verbose:
                    logger.info("Running command: %s", " ".join(dl_cmd))
                
                # Run without capturing output to show progress in real-time
                subprocess.run(dl_cmd, check=True, text=True)

                # Verify downloads by checking the directory
                downloaded_files = [
                    f for f in channel_dir.iterdir()
                    if f.is_file() and not f.name.endswith(".tmp") and f.name != "tdl-export.json"
                ]

                if not downloaded_files:
                    logger.warning(
                        "Download command completed but no files were found for channel %s", 
                        channel.name
                    )
                    return False

                logger.info(
                    "Successfully downloaded %d files for channel %s",
                    len(downloaded_files),
                    channel.name
                )
                return True

            except subprocess.CalledProcessError as e:
                error_msg = f"Download failed for channel {channel.name}"
                if hasattr(e, 'stderr') and e.stderr:
                    error_msg += f": {e.stderr}"
                logger.error(error_msg)
                return False

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to export channel {channel.name}"
            if e.stderr:
                error_msg += f": {e.stderr}"
            logger.error(error_msg)
            return False
        except Exception as e:
            logger.error("Unexpected error downloading channel %s: %s", channel.name, str(e))
            return False
        finally:
            # Clean up export file if it exists outside channel directory
            if Path("tdl-export.json").exists():
                Path("tdl-export.json").unlink()

    def download_channel_wrapper(self, channel: ChannelConfig) -> Tuple[ChannelConfig, bool, Optional[str]]:
        """
        Wrapper for download_channel to use with concurrent.futures.
        
        Args:
            channel: Channel configuration
            
        Returns:
            Tuple containing (channel, success_status, error_message)
        """
        try:
            logger.info("Downloading channel: %s", channel.name)
            success = self.download_channel(channel)
            if success:
                channel.working_dir = self.downloads_dir / channel.name
                if not channel.working_dir.exists() or not any(channel.working_dir.iterdir()):
                    return channel, False, f"No files found in {channel.working_dir}"
                return channel, True, None
            else:
                return channel, False, "No files downloaded"
        except Exception as e:
            return channel, False, str(e)

    def setup_working_directory(self, channel: ChannelConfig) -> bool:
        """
        Set up working directory for channel processing.

        Args:
            channel: Channel configuration

        Returns:
            bool: True if directory was set up with files, False otherwise
        """
        source_dir = Path(channel.channel)
        if not source_dir.exists() or not any(source_dir.iterdir()):
            return False

        # Create target directory if we have files to move
        target_dir = Path(channel.name)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(exist_ok=True)

        # Move downloaded files to target directory
        for item in source_dir.iterdir():
            self.safe_move(item, target_dir / item.name)
        source_dir.rmdir()  # Remove empty source directory

        channel.working_dir = target_dir
        return True

    def get_archive_files(self, directory: Path) -> List[Path]:
        """
        Get list of archive files in directory.

        Args:
            directory: Directory to search

        Returns:
            List of archive file paths
        """
        if not directory.exists():
            return []

        archive_extensions = {".zip", ".rar", ".7z"}
        return [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in archive_extensions
        ]

    def extract_single_archive(
        self, archive_path: Path, password: Optional[str] = None
    ) -> bool:
        """
        Extract a single archive file.

        Args:
            archive_path: Path to archive file
            password: Optional password for extraction

        Returns:
            bool: True if extraction was successful
        """
        # Default patterns for text files and stealer logs
        # Using *pattern* format to match the term anywhere in the filename
        # -ir! switch handles case-insensitive matching
        default_patterns = [
            "*.txt",
            "*.csv",
            "*pass*",
            "*auto*"
        ]
        
        try:
            # Base command - use just the filename since we're changing to the directory
            # Encode filename as bytes to handle non-ASCII characters
            base_cmd = ["7z", "x", str(archive_path.name), "-aoa"]

            # Add patterns
            for pattern in default_patterns:
                base_cmd.extend(["-ir!" + pattern])

            # Add password if provided
            if password:
                base_cmd.extend(["-p" + password])
                if self.verbose:
                    logger.info("Using password for %s: %s", archive_path.name, password)
            else:
                if self.verbose:
                    logger.info("No password provided for %s", archive_path.name)

            if self.verbose:
                logger.info("Running command: %s", " ".join(base_cmd))
                logger.info("Working directory: %s", archive_path.parent)

            # Set a shorter timeout for problematic archives
            timeout = self.settings.get("archive", "extract_timeout", default=300)  # 5 minutes default
            
            env = os.environ.copy()
            env["LANG"] = "C.UTF-8"  # Force UTF-8 encoding
            
            process = subprocess.Popen(
                base_cmd,
                cwd=archive_path.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                encoding='utf-8',
                errors='replace'  # Replace invalid chars instead of failing
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
                
                # Check for specific Unicode-related errors in stderr
                if stderr and ("ERROR: Cannot convert" in stderr or "Wrong password" in stderr):
                    if self.verbose:
                        logger.error(
                            "7z extraction failed for %s:\n%s",
                            archive_path.name,
                            stderr[:500]  # Limit error message length
                        )
                    return False
                
                if process.returncode != 0:
                    if self.verbose:
                        logger.error(
                            "7z extraction failed for %s:\n%s",
                            archive_path.name,
                            stderr[:500] if stderr else "No error message"
                        )
                        logger.error("Exit code: %d", process.returncode)
                    return False
                
                if self.verbose and stdout:
                    logger.info("7z output for %s:\n%s", archive_path.name, stdout)
                return True
                
            except subprocess.TimeoutExpired:
                self.terminate_process(process)
                logger.error(
                    "Extraction timed out after %d seconds for %s",
                    timeout,
                    archive_path.name,
                )
                return False

        except Exception as e:
            logger.error(
                "Failed to run extraction for %s: %s", archive_path.name, str(e)
            )
            return False

    def extract_archives(self, channel: ChannelConfig) -> bool:
        """
        Extract archives from channel directory.

        Args:
            channel: Channel configuration

        Returns:
            bool: True if any archives were successfully extracted
        """
        if not channel.working_dir or not channel.working_dir.exists():
            return False

        archive_files = self.get_archive_files(channel.working_dir)
        if not archive_files:
            return True  # No archives is considered success

        success_count = 0
        total_count = len(archive_files)

        for archive in archive_files:
            logger.info(
                "Processing archive %s for channel %s", archive.name, channel.name
            )

            # If password is specified in CSV, try only with password
            if channel.has_password:
                if self.verbose:
                    logger.info(
                        "Attempting extraction with password for %s", archive.name
                    )
                if self.extract_single_archive(archive, channel.password):
                    success_count += 1
                    logger.info("Successfully extracted %s with password", archive.name)
                else:
                    logger.warning(
                        "Failed to extract %s with password, skipping file",
                        archive.name,
                    )
            else:
                # No password in CSV, try without password
                if self.verbose:
                    logger.info(
                        "Attempting extraction without password for %s", archive.name
                    )
                if self.extract_single_archive(archive):
                    success_count += 1
                    logger.info("Successfully extracted %s without password", archive.name)
                else:
                    logger.warning(
                        "Failed to extract %s without password, skipping file",
                        archive.name,
                    )

        if success_count == 0:
            logger.error("Failed to extract any archives for channel %s", channel.name)
            return False

        if success_count < total_count:
            logger.warning(
                "Extracted %d out of %d archives for channel %s",
                success_count,
                total_count,
                channel.name,
            )
        else:
            logger.info(
                "Successfully extracted all %d archives for channel %s",
                total_count,
                channel.name,
            )

        return True

    def has_archives(self, directory: Path) -> bool:
        """
        Check if directory contains archive files.

        Args:
            directory: Directory to check

        Returns:
            bool: True if archives are present
        """
        return bool(self.get_archive_files(directory))

    def deduplicate(self, directory: Path) -> None:
        """
        Remove duplicate files from directory.

        Args:
            directory: Directory to deduplicate
        """
        if not directory or not directory.exists():
            return

        try:
            subprocess.run(
                ["rdfind", "-deleteduplicates", "true", "."],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed to deduplicate files: %s", e.stderr)
        finally:
            results_file = directory / "results.txt"
            if results_file.exists():
                results_file.unlink()

    def process_text_files(self, channel: ChannelConfig) -> bool:
        """
        Process text files from channel directory.

        Args:
            channel: Channel configuration

        Returns:
            bool: True if files were processed and non-empty results were generated
        """
        if not channel.working_dir or not channel.working_dir.exists():
            return False

        output_file = f"{channel.name}-{self.date_suffix}-combo.csv"

        try:
            # Only process files in the root directory, don't descend into subdirectories
            txt_files = list(channel.working_dir.glob("*.txt"))
            if not txt_files:
                logger.info("No text files found in root directory of %s", channel.name)
                return False

            # Filter out empty files
            non_empty_files = []
            for txt_file in txt_files:
                if txt_file.stat().st_size > 0:
                    non_empty_files.append(txt_file)
            
            if not non_empty_files:
                logger.info("No non-empty text files found in root directory of %s", channel.name)
                return False
            
            # Create a file list for cat command
            file_list_path = channel.working_dir / "file_list.tmp"
            with open(file_list_path, "w", encoding="utf-8") as f:
                for txt_file in non_empty_files:
                    f.write(f"{txt_file}\n")
            
            # Use cat to combine files instead of loading into memory
            combined_file = channel.working_dir / "combined.txt"
            cat_cmd = ["cat"]
            cat_cmd.extend([str(f) for f in non_empty_files])
            
            with open(combined_file, "w", encoding="utf-8") as outfile:
                subprocess.run(cat_cmd, stdout=outfile, check=True)
                
            # Clean up temp file
            file_list_path.unlink(missing_ok=True)
            
            # Check if combined file has content
            if combined_file.stat().st_size == 0:
                logger.info("No valid content found in text files for %s", channel.name)
                # Clean up empty combined file
                combined_file.unlink(missing_ok=True)
                return False

            # Sort and deduplicate
            sort_settings = self.settings.get("sort", default={})
            sort_cmd = [
                "sort",
                "-T", str(sort_settings.get("temp_dir", "/tmp")),
                "-u", "-b", "-i", "-f",
                f"-S{sort_settings.get('memory_percent', 30)}%",
                f"--parallel={sort_settings.get('max_parallel', 16)}",
                "-o", str(output_file),
                str(combined_file)
            ]
            
            # Set LC_ALL=C for consistent sorting
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            
            subprocess.run(
                sort_cmd, 
                check=True, 
                cwd=channel.working_dir,
                env=env
            )

            # Check if output file exists and has actual content
            output_path = channel.working_dir / output_file
            if not output_path.exists() or output_path.stat().st_size <= 1:  # Account for possible newline
                logger.info("No unique content found in %s", channel.name)
                output_path.unlink(missing_ok=True)
                return False

            # Move to output directory only if we have content
            self.safe_move(output_path, self.output_dir / output_file)
            
            # Clean up combined file
            combined_file.unlink(missing_ok=True)
            return True

        except (subprocess.CalledProcessError, OSError) as e:
            logger.error(
                "Failed to process text files for %s: %s", channel.name, str(e)
            )
            # Clean up any partial files
            (channel.working_dir / "combined.txt").unlink(missing_ok=True)
            (channel.working_dir / output_file).unlink(missing_ok=True)
            return False

    def process_stealer_logs(self, channel: ChannelConfig) -> bool:
        """
        Process stealer logs from channel directory.

        Args:
            channel: Channel configuration

        Returns:
            bool: True if files were processed and non-empty results were generated
        """
        if not channel.working_dir or not channel.working_dir.exists():
            return False

        if not self.has_archives(channel.working_dir):
            return False

        try:
            # Run stealer log processor
            processor_path = self.settings.get("stealer_log_processor", "path")
            if not processor_path:
                logger.error("Stealer log processor path not configured")
                return False

            subprocess.run(
                ["python3", processor_path, str(channel.working_dir)],
                check=True,
                capture_output=True,
                text=True,
            )

            # Move and rename non-empty result files
            has_results = False
            for file in ["credentials.csv", "autofills.csv"]:
                source = channel.working_dir / file
                if source.exists():
                    # Check if file has actual content
                    if source.stat().st_size > 1:  # More than just a newline
                        new_name = f"{channel.name}-{self.date_suffix}-{file}"
                        self.safe_move(source, self.output_dir / new_name)
                        has_results = True
                    else:
                        # Remove empty result file
                        source.unlink()

            return has_results

        except subprocess.CalledProcessError as e:
            logger.error(
                "Failed to process stealer logs for %s: %s", channel.name, e.stderr
            )
            return False
        except OSError as e:
            logger.error("Failed to move result files for %s: %s", channel.name, str(e))
            return False

    def process(self) -> None:
        """Process all channels."""
        self.load_channels()
        channels_to_cleanup = []
        successful_downloads = []

        # First, download all channels if not in process-only mode
        if not self.process_only:
            logger.info("Starting sequential channel downloads")
            
            # Process channels sequentially instead of in parallel
            # because TDL can't handle multiple channel downloads at once (database lock issues)
            for channel in self.channels:
                channel_result, success, error_msg = self.download_channel_wrapper(channel)
                if success:
                    successful_downloads.append(channel_result)
                    logger.info("Successfully downloaded files for channel: %s", channel_result.name)
                else:
                    logger.info("Skipping channel %s - %s", channel.name, error_msg)

            if not successful_downloads:
                logger.warning("No files were downloaded from any channel")
                return
            
            # Deduplicate across all downloaded channels to handle duplicates across channels
            logger.info("Deduplicating files across all channels")
            self.deduplicate(self.downloads_dir)
        else:
            # In process-only mode, check for existing files in downloads directory
            logger.info("Running in process-only mode, checking for existing files")
            for channel in self.channels:
                channel.working_dir = self.downloads_dir / channel.name
                if channel.working_dir.exists() and any(channel.working_dir.iterdir()):
                    successful_downloads.append(channel)
                    logger.info("Found existing files for channel: %s", channel.name)
                else:
                    logger.info("No existing files found for channel: %s", channel.name)

            if not successful_downloads:
                logger.warning("No existing files found for any channel")
                return

        # Process channels in parallel (this part is still parallel because it doesn't use TDL)
        max_workers = min(len(successful_downloads), os.cpu_count() or 4)
        logger.info(f"Starting parallel processing with {max_workers} workers")
        
        # Function for parallel processing
        def process_channel(channel):
            try:
                logger.info("Processing downloaded files for channel: %s", channel.name)
                
                # Process the channel
                extraction_success = self.extract_archives(channel)
                
                # Deduplicate after extraction before further processing
                if extraction_success and channel.working_dir and channel.working_dir.exists():
                    logger.info("Deduplicating extracted files for %s", channel.name)
                    self.deduplicate(channel.working_dir)
                    
                processing_success = False

                # Try processing stealer logs if we have archives
                if extraction_success and self.has_archives(channel.working_dir):
                    stealer_success = self.process_stealer_logs(channel)
                    if stealer_success:
                        processing_success = True
                        logger.info("Successfully processed stealer logs for %s", channel.name)

                # Always try processing text files regardless of archive status
                text_success = self.process_text_files(channel)
                if text_success:
                    processing_success = True
                    logger.info("Successfully processed text files for %s", channel.name)

                if processing_success:
                    return channel, True
                else:
                    logger.warning("No valid results found for %s", channel.name)
                    return channel, False
            except Exception as e:
                logger.error("Error processing %s: %s", channel.name, str(e))
                return channel, False
        
        # Process channels in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_channel = {
                executor.submit(process_channel, channel): channel 
                for channel in successful_downloads
            }
            
            for future in concurrent.futures.as_completed(future_to_channel):
                try:
                    channel, success = future.result()
                    if success:
                        channels_to_cleanup.append(channel)
                except Exception as e:
                    logger.error("Unexpected error in parallel processing: %s", str(e))

        # Cleanup phase after all processing is done
        if channels_to_cleanup:
            cleanup_all = True
            if not self.auto_clean:
                cleanup_all = (
                    input(
                        "Do you want to clean up all processed directories? (y/n): "
                    ).lower()
                    == "y"
                )
            if cleanup_all:
                for channel in channels_to_cleanup:
                    try:
                        if channel.working_dir and channel.working_dir.exists():
                            shutil.rmtree(channel.working_dir)
                            logger.info("Cleaned up %s directory", channel.name)
                    except OSError as e:
                        logger.error(
                            "Failed to clean up %s directory: %s", channel.name, str(e)
                        )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Telegram Channel Processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, help="Input CSV file with channel configurations"
    )
    parser.add_argument("--start", required=True, help="Start date (DD-MM-YYYY)")
    parser.add_argument("--end", required=True, help="End date (DD-MM-YYYY)")
    parser.add_argument("--output-dir", help="Output directory for processed files")
    parser.add_argument("--download-dir", help="Directory for downloaded files")
    parser.add_argument("--settings", help="Path to settings JSON file")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output including tdl commands",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Skip download phase and only process existing files",
    )
    parser.add_argument(
        "--auto-clean",
        action="store_true",
        help="Automatically clean up after processing without prompting",
    )

    args = parser.parse_args()

    try:
        # Convert dates to epoch
        try:
            start_epoch = datetime.strptime(args.start, "%d-%m-%Y").timestamp()
            end_epoch = datetime.strptime(args.end, "%d-%m-%Y").timestamp()
        except ValueError as e:
            logger.error("Invalid date format. Use DD-MM-YYYY")
            sys.exit(1)

        processor = TelegramProcessor(
            input_file=args.input,
            start_date=str(int(start_epoch)),
            end_date=str(int(end_epoch)),
            output_dir=args.output_dir,
            download_dir=args.download_dir,
            settings_file=args.settings,
            verbose=args.verbose,
            process_only=args.process_only,
            auto_clean=args.auto_clean,
        )
        
        # Check dependencies before starting
        processor.check_dependencies()

        processor.process()
    except ProcessingError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        sys.exit(1)
    finally:
        processor.cleanup_temp_files()


if __name__ == "__main__":
    main()
