from __future__ import annotations

import os
from concurrent.futures import as_completed

from .logging_utils import logger

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class WorkflowMixin:
    """Top-level orchestration for multi-channel processing."""

    def process(self) -> None:
        self.load_channels()
        self.load_checkpoint()

        channels_to_cleanup = []
        successful_downloads = []

        if not self.process_only:
            logger.info("Starting sequential channel downloads")
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

            logger.info("Deduplicating files across all channels")
            self.deduplicate(self.downloads_dir)
        else:
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

        max_workers = self.settings.get("processing", "max_workers", default=None) or min(
            len(successful_downloads), os.cpu_count() or 4
        )
        logger.info("Starting parallel processing with %s workers", max_workers)

        with self.process_executor_cls(max_workers=max_workers) as executor:
            future_to_channel = {executor.submit(self.process_channel_files, channel): channel for channel in successful_downloads}
            futures = tqdm(as_completed(future_to_channel), total=len(future_to_channel), desc="Processing") if TQDM_AVAILABLE else as_completed(future_to_channel)
            for future in futures:
                try:
                    channel, success = future.result()
                    if success:
                        channels_to_cleanup.append(channel)
                except Exception as exc:
                    channel = future_to_channel[future]
                    logger.error("Unexpected error processing %s: %s", channel.name, str(exc))

        logger.info("All processing completed, preparing for cleanup")
        if channels_to_cleanup:
            cleanup_all = True
            if not self.auto_clean:
                cleanup_all = (
                    self.system.prompt(
                        f"Do you want to clean up {len(channels_to_cleanup)} processed directories? (y/n): "
                    ).lower()
                    == "y"
                )
            if cleanup_all:
                for channel in channels_to_cleanup:
                    try:
                        if channel.working_dir and channel.working_dir.exists():
                            self.remove_tree(channel.working_dir)
                            logger.info("Cleaned up %s directory", channel.name)
                    except OSError as exc:
                        logger.error("Failed to clean up %s directory: %s", channel.name, str(exc))

        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
