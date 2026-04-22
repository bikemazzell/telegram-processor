#!/usr/bin/env python3

"""CLI entrypoint for the Telegram channel processor."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from telegram_processor_lib import (
    ChannelConfig,
    ProcessingError,
    Settings,
    SystemOps,
    TQDM_AVAILABLE,
    TelegramProcessor,
    setup_logging,
)

__all__ = [
    "ChannelConfig",
    "ProcessingError",
    "Settings",
    "SystemOps",
    "TQDM_AVAILABLE",
    "TelegramProcessor",
    "main",
    "setup_logging",
]


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Telegram Channel Processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input CSV file with channel configurations")
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
        settings_file = Path(args.settings or "settings.json")
        if settings_file.exists():
            with open(settings_file, "r", encoding="utf-8") as handle:
                settings_data = json.load(handle)
        else:
            settings_data = None
    except Exception:
        settings_data = None

    logger = setup_logging(args.verbose, settings_data)

    if not TQDM_AVAILABLE:
        logger.info("tqdm not available. Install it for progress bars: pip install tqdm")

    try:
        try:
            start_epoch = datetime.strptime(args.start, "%d-%m-%Y").timestamp()
            end_epoch = datetime.strptime(args.end, "%d-%m-%Y").timestamp()
        except ValueError:
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
        processor.check_dependencies()
        processor.process()
    except ProcessingError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception:
        logger.exception("Unexpected error occurred")
        sys.exit(1)
    finally:
        if "processor" in locals():
            processor.cleanup_temp_files()


if __name__ == "__main__":
    main()
