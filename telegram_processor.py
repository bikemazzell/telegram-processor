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
import logging.handlers
import os
import shutil
import subprocess
import sys
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Configure logging with rotation
def setup_logging(verbose: bool = False, settings: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Setup logging with file rotation and console output."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation - use settings if available
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get logging settings
    if settings:
        max_size = settings.get("logging", {}).get("max_file_size_mb", 10) * 1024 * 1024
        backup_count = settings.get("logging", {}).get("backup_count", 5)
    else:
        max_size = 10 * 1024 * 1024  # 10MB default
        backup_count = 5
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "telegram_processor.log",
        maxBytes=max_size,
        backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger (will be reconfigured in main)
logger = logging.getLogger(__name__)

# Check if tqdm is available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Don't log warning here as logger may not be configured yet


class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass


class ChannelConfig:
    """Configuration for a Telegram channel."""

    def __init__(self, name: str, channel: str, passwords: Optional[List[str]] = None) -> None:
        """
        Initialize channel configuration.

        Args:
            name: Display name for the channel
            channel: Channel identifier (ID or username)
            passwords: Optional list of passwords for archive extraction

        Raises:
            ValueError: If name or channel is empty
        """
        if not name or not name.strip():
            raise ValueError("Channel name cannot be empty")
        if not channel or not channel.strip():
            raise ValueError("Channel identifier cannot be empty")

        self.name: str = name.strip()
        self.channel: str = channel.strip()
        self.passwords: List[str] = [p.strip() for p in passwords if p and p.strip()] if passwords else []
        self.working_dir: Optional[Path] = None

    @property
    def has_passwords(self) -> bool:
        """Check if channel has any valid passwords."""
        return bool(self.passwords)


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
                "max_retries": 3,
                "retry_delay": 5,
            },
            "sort": {
                "memory_percent": 30,
                "max_parallel": 16,
                "temp_dir": "/tmp",
                "chunk_size": 1000000,  # lines per chunk for external sort
            },
            "archive": {
                "extract_patterns": [
                    "*.txt",
                    "*.csv",
                    "*pass*",
                    "*auto*"
                ],
                "supported_extensions": [".zip", ".rar", ".7z"],
                "extract_timeout": 300,  # 5 minutes default
                "max_retries": 2,
                "retry_delay": 2,
            },
            "processing": {
                "max_workers": None,  # None means use CPU count
                "checkpoint_interval": 100,  # Save progress every N files
                "min_file_size_bytes": 0,
                "min_result_file_size_bytes": 1,
            },
            "logging": {
                "max_file_size_mb": 10,
                "backup_count": 5,
                "temp_cleanup_patterns": [
                    "tdl-export*.json",
                    "*.temp",
                    "sort*"
                ]
            },
            "subprocess": {
                "default_timeout": 300,
                "terminate_timeout": 5
            }
        }

    def get(self, *keys, default=None):
        """
        Get a nested setting value.

        Args:
            *keys: Sequence of keys to access nested settings
            default: Default value if key doesn't exist

        Returns:
            Setting value or default
        """
        try:
            current = self.settings
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def prepare_extensions(self, extensions: List[str]) -> List[str]:
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


class RetryMixin:
    """Mixin class for retry functionality."""
    
    def retry_operation(self, func, *args, max_retries: int = 3, retry_delay: int = 5, **kwargs):
        """
        Retry an operation with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Positional arguments for func
            max_retries: Maximum number of retries
            retry_delay: Initial delay between retries in seconds
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of successful operation
            
        Raises:
            Last exception if all retries fail
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %d seconds...",
                    attempt + 1, max_retries, str(e), wait_time
                )
                time.sleep(wait_time)


class TelegramProcessor(RetryMixin):
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
        
        # Track processing state
        self.processed_channels: Dict[str, bool] = {}
        self.checkpoint_file = self.downloads_dir / ".processing_checkpoint.json"

    def check_dependencies(self) -> None:
        """
        Check if required external tools are available.

        Raises:
            ProcessingError: If any required tool is missing
        """
        required_tools = {
            "tdl": "Telegram downloader (tdl) - Install from: https://github.com/iyear/tdl",
            "7z": "7-Zip archiver - Install: apt-get install p7zip-full",
            "rdfind": "Duplicate file finder - Install: apt-get install rdfind",
        }
        
        missing_tools = []
        for tool, description in required_tools.items():
            try:
                subprocess.run(
                    ["which", tool], 
                    check=True, 
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError:
                missing_tools.append(f"{tool}: {description}")
        
        if missing_tools:
            error_msg = "Missing required tools:\n" + "\n".join(missing_tools)
            raise ProcessingError(error_msg)

    def load_checkpoint(self) -> None:
        """Load processing checkpoint if it exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    self.processed_channels = json.load(f)
                logger.info("Loaded checkpoint with %d processed channels", len(self.processed_channels))
            except Exception as e:
                logger.warning("Failed to load checkpoint: %s", e)
                self.processed_channels = {}

    def save_checkpoint(self) -> None:
        """Save processing checkpoint."""
        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.processed_channels, f)
        except Exception as e:
            logger.warning("Failed to save checkpoint: %s", e)

    def load_channels(self) -> None:
        """
        Load channel configurations from CSV file.

        Raises:
            ProcessingError: If CSV file is invalid or missing required columns
        """
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    raise ProcessingError("CSV file is empty")

                # Basic header validation
                if len(header) < 2 or header[0].strip().lower() != 'name' or header[1].strip().lower() != 'channel':
                    raise ProcessingError("CSV must start with 'name' and 'channel' columns")

                for row in reader:
                    if not row or not row[0] or not row[1]:
                        continue  # Skip empty rows or rows missing name/channel

                    name = row[0]
                    channel_id = row[1]
                    passwords = row[2:] if len(row) > 2 else []

                    channel = ChannelConfig(
                        name=name,
                        channel=channel_id,
                        passwords=passwords,
                    )
                    self.channels.append(channel)

            if not self.channels:
                raise ProcessingError("No channels found in CSV file")

        except csv.Error as e:
            raise ProcessingError(f"Failed to parse CSV file: {e}")

    def cleanup_temp_files(self) -> None:
        """Clean up any temporary files."""
        try:
            # Get cleanup patterns from settings with fallback
            cleanup_patterns = self.settings.get("logging", "temp_cleanup_patterns", default=[
                "tdl-export*.json", "*.temp", "sort*"
            ])
            
            # Ensure we have a list to iterate over
            if cleanup_patterns is None:
                cleanup_patterns = ["tdl-export*.json", "*.temp", "sort*"]
            
            # Clean up files in current directory
            for pattern in cleanup_patterns:
                if pattern != "sort*":  # Handle sort files separately
                    for temp_file in Path().glob(pattern):
                        temp_file.unlink(missing_ok=True)
            
            # Clean up sort temp files in temp directory
            temp_dir = Path(self.settings.get("sort", "temp_dir", default="/tmp"))
            for sort_temp in temp_dir.glob("sort*"):
                if sort_temp.is_file():
                    sort_temp.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to clean up temporary files: %s", e)

    @contextmanager
    def managed_process(self, *args, **kwargs):
        """Context manager for subprocess with proper cleanup."""
        process = subprocess.Popen(*args, **kwargs)
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
        """
        Safely move a file with proper error handling.

        Args:
            src: Source path
            dst: Destination path

        Raises:
            ProcessingError: If move operation fails
        """
        try:
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except Exception as e:
            raise ProcessingError(f"Failed to move {src} to {dst}: {e}")

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
                
            # Extract file information from messages
            files = []
            for message in export_data.get("messages", []):
                if message.get("type") == "message" and "file" in message:
                    files.append(message["file"])
            return files
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(
                "Failed to parse export file for channel %s: %s", channel.name, str(e)
            )
            return []

    def download_channel_with_retry(self, channel: ChannelConfig) -> bool:
        """
        Download content from a Telegram channel with retry logic.

        Args:
            channel: Channel configuration

        Returns:
            bool: True if any files were downloaded, False otherwise
        """
        max_retries = self.settings.get("tdl", "max_retries", default=3)
        retry_delay = self.settings.get("tdl", "retry_delay", default=5)
        
        return self.retry_operation(
            self._download_channel_impl,
            channel,
            max_retries=max_retries,
            retry_delay=retry_delay
        )

    def _download_channel_impl(self, channel: ChannelConfig) -> bool:
        """
        Implementation of channel download logic.

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
            
            # Capture output to prevent interference with tqdm
            export_result = subprocess.run(export_cmd, check=True, capture_output=True, text=True, errors='replace')
            if self.verbose and export_result.stderr:
                # tdl sends progress and info to stderr
                logger.info("tdl export output for %s:\n%s", channel.name, export_result.stderr.strip())

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
                excluded_extensions = self.settings.prepare_extensions(excluded_extensions)
                dl_cmd.extend(["-e", ",".join(excluded_extensions)])

            # Run download with proper error handling
            if self.verbose:
                logger.info("Running command: %s", " ".join(dl_cmd))
            
            # Capture output to prevent interference with tqdm
            dl_result = subprocess.run(dl_cmd, check=True, capture_output=True, text=True, errors='replace')
            if self.verbose and dl_result.stderr:
                # tdl sends progress and info to stderr
                logger.info("tdl download output for %s:\n%s", channel.name, dl_result.stderr.strip())

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
            error_msg = f"Failed to process channel {channel.name}"
            if hasattr(e, 'stderr') and e.stderr:
                error_msg += f": {e.stderr}"
            logger.error(error_msg)
            raise
        except Exception as e:
            logger.error("Unexpected error downloading channel %s: %s", channel.name, str(e))
            raise
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
            # Check if already processed
            if self.processed_channels.get(channel.name, False):
                logger.info("Channel %s already processed, skipping", channel.name)
                channel.working_dir = self.downloads_dir / channel.name
                return channel, True, None
                
            success = self.download_channel_with_retry(channel)
            if success:
                channel.working_dir = self.downloads_dir / channel.name
                if not channel.working_dir.exists() or not any(channel.working_dir.iterdir()):
                    return channel, False, f"No files found in {channel.working_dir}"
                # Mark as processed
                self.processed_channels[channel.name] = True
                self.save_checkpoint()
                return channel, True, None
            else:
                return channel, False, "No files downloaded"
        except Exception as e:
            return channel, False, str(e)

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

        # Get supported archive extensions from settings
        supported_extensions = self.settings.get("archive", "supported_extensions", default=[".zip", ".rar", ".7z"])
        archive_extensions = {ext.lower() for ext in supported_extensions}
        return [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in archive_extensions
        ]

    def extract_single_archive(
        self, archive_path: Path, password: Optional[str] = None
    ) -> bool:
        """
        Extract a single archive file with retry logic.

        Args:
            archive_path: Path to archive file
            password: Optional password for extraction

        Returns:
            bool: True if extraction was successful
        """
        max_retries = self.settings.get("archive", "max_retries", default=2)
        
        for attempt in range(max_retries):
            try:
                if self._extract_archive_impl(archive_path, password):
                    return True
                if attempt < max_retries - 1:
                    retry_delay = self.settings.get("archive", "retry_delay", default=2)
                    logger.warning(
                        "Extraction failed for %s, retrying (%d/%d) in %d seconds...",
                        archive_path.name, attempt + 1, max_retries, retry_delay
                    )
                    time.sleep(retry_delay)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        "Failed to extract %s after %d attempts: %s",
                        archive_path.name, max_retries, str(e)
                    )
                    return False
        return False

    def _extract_archive_impl(
        self, archive_path: Path, password: Optional[str] = None
    ) -> bool:
        """
        Implementation of archive extraction logic.

        Args:
            archive_path: Path to archive file
            password: Optional password for extraction

        Returns:
            bool: True if extraction was successful
        """
        patterns = self.settings.get("archive", "extract_patterns", default=["*.txt", "*.csv"])
        
        try:
            # Base command
            base_cmd = ["7z", "x", str(archive_path.name), "-aoa"]

            # Add patterns
            for pattern in patterns:
                base_cmd.extend(["-ir!" + pattern])

            # Add password if provided
            if password:
                base_cmd.extend(["-p" + password])

            if self.verbose:
                logger.info("Running command: %s", " ".join(base_cmd))
                logger.info("Working directory: %s", archive_path.parent)

            # Set timeout
            timeout = self.settings.get("archive", "extract_timeout", default=300)
            
            env = os.environ.copy()
            env["LANG"] = "C.UTF-8"
            
            with self.managed_process(
                base_cmd,
                cwd=archive_path.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                encoding='utf-8',
                errors='replace'
            ) as process:
                try:
                    _, stderr = process.communicate(timeout=timeout)
                    
                    # Check for specific errors
                    if stderr and ("ERROR: Cannot convert" in stderr or "Wrong password" in stderr):
                        if self.verbose:
                            logger.error(
                                "7z extraction failed for %s:\n%s",
                                archive_path.name,
                                stderr[:500]
                            )
                        return False
                    
                    if process.returncode != 0:
                        if self.verbose:
                            logger.error(
                                "7z extraction failed for %s with exit code %d",
                                archive_path.name,
                                process.returncode
                            )
                        return False
                    
                    return True
                    
                except subprocess.TimeoutExpired:
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
        Extract archives from channel directory. Tries all provided passwords,
        and if all fail, attempts extraction without a password.

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

            extracted = False
            # Try extraction with passwords if provided
            if channel.has_passwords:
                if self.verbose:
                    logger.info(
                        "Attempting extraction with %d password(s) for %s",
                        len(channel.passwords),
                        archive.name,
                    )
                for password in channel.passwords:
                    if self.extract_single_archive(archive, password):
                        logger.info("Successfully extracted %s with a provided password", archive.name)
                        extracted = True
                        break  # Move to the next archive

            # If not extracted (either no passwords or passwords failed), try without a password
            if not extracted:
                if self.verbose:
                    logger.info("Attempting extraction without password for %s", archive.name)
                if self.extract_single_archive(archive):
                    logger.info("Successfully extracted %s without a password", archive.name)
                    extracted = True

            if extracted:
                success_count += 1
            else:
                logger.warning("Failed to extract %s with any method", archive.name)

        if success_count == 0 and total_count > 0:
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
            logger.info("Deduplication completed for %s", directory)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to deduplicate files: %s", e.stderr)
        finally:
            results_file = directory / "results.txt"
            if results_file.exists():
                results_file.unlink()

    def process_text_files_streaming(self, channel: ChannelConfig) -> bool:
        """
        Process text files from channel directory using streaming approach.

        Args:
            channel: Channel configuration

        Returns:
            bool: True if files were processed and non-empty results were generated
        """
        if not channel.working_dir or not channel.working_dir.exists():
            return False

        output_file = f"{channel.name}-{self.date_suffix}-combo.csv"

        try:
            # Only process files in the root directory
            txt_files = list(channel.working_dir.glob("*.txt"))
            if not txt_files:
                logger.info("No text files found in root directory of %s", channel.name)
                return False

            # Filter out empty files
            min_file_size = self.settings.get("processing", "min_file_size_bytes", default=0)
            non_empty_files = []
            for txt_file in txt_files:
                if txt_file.stat().st_size > min_file_size:
                    non_empty_files.append(txt_file)
            
            if not non_empty_files:
                logger.info("No non-empty text files found in root directory of %s", channel.name)
                return False
            
            # Use external sort for memory efficiency
            sort_settings = self.settings.get("sort", default={})
            
            # Create a temporary file for combined content
            with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                           dir=channel.working_dir, 
                                           suffix='.txt') as temp_combined:
                temp_combined_path = Path(temp_combined.name)
                
                # Stream files to temporary file
                for txt_file in non_empty_files:
                    try:
                        with open(txt_file, 'r', encoding='utf-8', errors='replace') as infile:
                            for line in infile:
                                temp_combined.write(line)
                    except Exception as e:
                        logger.warning("Error reading %s: %s", txt_file, e)
                        continue
            
            # Check if combined file has content
            if temp_combined_path.stat().st_size == 0:
                logger.info("No valid content found in text files for %s", channel.name)
                temp_combined_path.unlink(missing_ok=True)
                return False

            # Sort and deduplicate using external sort
            sort_cmd = [
                "sort",
                "-T", str(sort_settings.get("temp_dir", "/tmp")),
                "-u",  # unique
                "-S", f"{sort_settings.get('memory_percent', 30)}%",
                "--parallel", str(sort_settings.get('max_parallel', 16)),
                "-o", str(output_file),
                str(temp_combined_path)
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

            # Clean up temporary file
            temp_combined_path.unlink(missing_ok=True)

            # Check if output file exists and has actual content
            min_result_size = self.settings.get("processing", "min_result_file_size_bytes", default=1)
            output_path = channel.working_dir / output_file
            if not output_path.exists() or output_path.stat().st_size <= min_result_size:
                logger.info("No unique content found in %s", channel.name)
                output_path.unlink(missing_ok=True)
                return False

            # Move to output directory only if we have content
            self.safe_move(output_path, self.output_dir / output_file)
            
            return True

        except (subprocess.CalledProcessError, OSError) as e:
            logger.error(
                "Failed to process text files for %s: %s", channel.name, str(e)
            )
            # Clean up any partial files
            if 'temp_combined_path' in locals():
                temp_combined_path.unlink(missing_ok=True)
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
            processor_path = self.settings.get("stealer_log_processor", "path", default=None)
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
                    min_result_size = self.settings.get("processing", "min_result_file_size_bytes", default=1)
                    if source.stat().st_size > min_result_size:
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

    def process_channel_files(self, channel: ChannelConfig) -> Tuple[ChannelConfig, bool]:
        """
        Process files for a single channel.
        
        Args:
            channel: Channel configuration
            
        Returns:
            Tuple of (channel, success)
        """
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
            text_success = self.process_text_files_streaming(channel)
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

    def process(self) -> None:
        """Process all channels."""
        self.load_channels()
        self.load_checkpoint()
        
        channels_to_cleanup = []
        successful_downloads = []

        # First, download all channels if not in process-only mode
        if not self.process_only:
            logger.info("Starting sequential channel downloads")
            
            # Process channels sequentially for downloads
            # Use progress bar if available
            iterator = tqdm(self.channels, desc="Downloading") if TQDM_AVAILABLE else self.channels
            
            for channel in iterator:
                channel_result, success, error_msg = self.download_channel_wrapper(channel)
                if success:
                    successful_downloads.append(channel_result)
                    logger.info("Successfully downloaded files for channel: %s", channel_result.name)
                else:
                    logger.info("Skipping channel %s - %s", channel.name, error_msg)

            if not successful_downloads:
                logger.warning("No files were downloaded from any channel")
                return
            
            # Deduplicate across all downloaded channels
            logger.info("Deduplicating files across all channels")
            self.deduplicate(self.downloads_dir)
        else:
            # In process-only mode, check for existing files
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

        # Process channels in parallel using ProcessPoolExecutor for CPU-bound tasks
        max_workers = self.settings.get("processing", "max_workers", default=None) or min(len(successful_downloads), os.cpu_count() or 4)
        logger.info(f"Starting parallel processing with {max_workers} workers")
        
        # Use ProcessPoolExecutor for CPU-bound processing tasks
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_channel = {
                executor.submit(self.process_channel_files, channel): channel 
                for channel in successful_downloads
            }
            
            # Use progress bar if available
            futures = tqdm(as_completed(future_to_channel), 
                          total=len(future_to_channel), 
                          desc="Processing") if TQDM_AVAILABLE else as_completed(future_to_channel)
            
            for future in futures:
                try:
                    channel, success = future.result()
                    if success:
                        channels_to_cleanup.append(channel)
                except Exception as e:
                    channel = future_to_channel[future]
                    logger.error("Unexpected error processing %s: %s", channel.name, str(e))

        # Wait for all processing to complete before cleanup
        logger.info("All processing completed, preparing for cleanup")
        
        # Cleanup phase after all processing is done
        if channels_to_cleanup:
            cleanup_all = True
            if not self.auto_clean:
                cleanup_all = (
                    input(
                        f"Do you want to clean up {len(channels_to_cleanup)} processed directories? (y/n): "
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
                        
        # Clean up checkpoint file
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


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
    
    # Load settings first for logging configuration
    try:
        settings_file = Path(args.settings or "settings.json")
        if settings_file.exists():
            with open(settings_file, "r", encoding="utf-8") as f:
                settings_data = json.load(f)
        else:
            settings_data = None
    except Exception:
        settings_data = None
    
    # Setup logging based on verbosity and settings
    global logger
    logger = setup_logging(args.verbose, settings_data)
    
    # Log tqdm availability after logger is configured
    if not TQDM_AVAILABLE:
        logger.info("tqdm not available. Install it for progress bars: pip install tqdm")

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
        if 'processor' in locals():
            processor.cleanup_temp_files()


if __name__ == "__main__":
    main()