# Telegram Channel Processor

A Python tool for downloading and processing content from Telegram channels. This script automates the retrieval of files from channels, handling of archives, extraction of text content, and processing of stealer logs.

## Features

- Parallel downloading from multiple Telegram channels
- Archive extraction with password support
- Deduplication of files across channels and within extracted content
- Text file processing and combination
- Stealer log processing
- Memory-efficient processing of large files

## Requirements

### Python Dependencies
- Python 3.6+
- Dependencies listed in requirements.txt

### External Dependencies
The script requires the following external tools to be installed and available in your PATH:

- `tdl` - Telegram Downloader CLI tool
- `7z` - 7-Zip archive extraction tool
- `rdfind` - Tool for finding duplicate files
- `sort` - GNU sort utility (included in most Linux distributions)

## Installation

1. Clone this repository
2. Install the required Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install the required external tools according to your operating system
4. Configure your settings.json file (see Configuration section)

## Usage

```
python telegram_processor.py --input <input_file.csv> --start <DD-MM-YYYY> --end <DD-MM-YYYY> [options]
```

### Required Arguments

- `--input`: Path to CSV file with channel configurations
- `--start`: Start date in DD-MM-YYYY format
- `--end`: End date in DD-MM-YYYY format

### Optional Arguments

- `--output-dir`: Output directory for processed files (default: ./output)
- `--download-dir`: Directory for downloaded files (default: ./downloads)
- `--settings`: Path to settings JSON file (default: ./settings.json)
- `--verbose`: Show detailed output including tdl commands
- `--process-only`: Skip download phase and only process existing files
- `--auto-clean`: Automatically clean up after processing without prompting

### Input CSV Format

The input CSV file must contain the following columns:
- `name`: Display name for the channel
- `channel`: Channel identifier (ID or username)
- `password`: (Optional) Password for archive extraction

Example:
```
name,channel,password
Channel1,@channel1,password123
Channel2,@channel2,
```

## Configuration (settings.json)

The `settings.json` file controls various aspects of the script's behavior:

```json
{
    "stealer_log_processor": {
        "path": "/path/to/stealer-log-processor/main.py"
    },
    "tdl": {
        "max_parallel_downloads": 4,
        "reconnect_timeout": 0,
        "threads": 4,
        "bandwidth_limit": 0,
        "chunk_size": 128,
        "excluded_extensions": [
            "jpg", "gif", "png", "webp", "webm", "mp4"
        ],
        "included_extensions": [
            "zip", "rar", "7z", "txt", "csv"
        ]
    },
    "sort": {
        "memory_percent": 30,
        "max_parallel": 16,
        "temp_dir": "/tmp"
    },
    "archive": {
        "extract_patterns": [
            "*.txt",
            "*.csv",
            "*pass*",
            "*auto*"
        ],
        "extract_timeout": 3600
    }
}
```

### Configuration Sections

#### stealer_log_processor
- `path`: Path to the stealer log processor script

#### tdl
- `max_parallel_downloads`: Maximum number of parallel downloads
- `reconnect_timeout`: Timeout for reconnecting to Telegram
- `threads`: Thread count for export operations
- `bandwidth_limit`: Limit bandwidth usage in KiB/s (0 means unlimited)
- `excluded_extensions`: File extensions to skip when downloading (e.g. jpg, gif, png, webp, webm, mp4)

#### sort
- `memory_percent`: Percentage of memory to use for sorting
- `max_parallel`: Maximum parallel sort threads
- `temp_dir`: Temporary directory for sort operations

#### archive
- `extract_patterns`: Patterns for files to extract from archives
- `extract_timeout`: Timeout (seconds) for extraction operations

## Processing Workflow

1. **Download Phase**: The script downloads files from the specified Telegram channels sequentially
2. **Deduplication**: Files are deduplicated across all channels 
3. **Extraction**: Archive files are extracted, respecting provided passwords
4. **Post-Extraction Deduplication**: Files are deduplicated after extraction
5. **Processing**: 
   - Text files are combined, sorted, and deduplicated
   - Stealer logs are processed if archives are present
6. **Output**: Processed files are moved to the output directory
7. **Cleanup**: Temporary directories are removed (with user confirmation or auto-clean)

## Output Files

The script generates the following types of output files:
- `{channel_name}-{month-year}-combo.csv`: Combined and deduplicated text content (typically UPL format)
- `{channel_name}-{month-year}-credentials.csv`: Processed credentials (from stealer logs)
- `{channel_name}-{month-year}-autofills.csv`: Processed autofill data (from stealer logs)

All output files are stored in the specified output directory. 