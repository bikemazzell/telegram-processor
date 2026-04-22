# Telegram Channel Processor

Python CLI for harvesting files from Telegram channels, extracting archives, deduplicating results, and producing channel-scoped outputs. It supports both channels with known static passwords and funnel channels where the archive password is written in the Telegram post itself.

## Features

- Sequential channel download workflow using `tdl`
- Archive extraction with `7z`
- Static passwords via `password`, `password1`, `password2`, and similar CSV columns
- Post-derived passwords via `password_source=password_in_post`
- Same-message password matching for funnel channels
- Cross-channel and per-channel deduplication with `rdfind`
- Text aggregation and sorted unique combo output
- Optional stealer-log processing into `credentials.csv` and `autofills.csv`
- Channel validation mode that can comment out inactive CSV rows

## Requirements

### Python

- Python 3.10+
- Optional: `tqdm` for progress bars

### External tools

- `tdl`
- `7z`
- `rdfind`
- `sort`

In normal processing mode all four are required. In `--process-only` mode, `tdl` is not required because the script only processes files already present in the downloads directory.

## Usage

### Process channels

```bash
python3 telegram_processor.py \
  --input channels.csv \
  --start 01-04-2026 \
  --end 22-04-2026
```

Useful options:

- `--output-dir`: output directory for final results
- `--download-dir`: directory for downloaded and extracted channel files
- `--settings`: path to `settings.json`
- `--verbose`: log `tdl` and extraction command details
- `--process-only`: skip Telegram downloads and process files already in `--download-dir`
- `--auto-clean`: remove processed channel directories without prompting

### Check channels only

```bash
python3 telegram_processor.py \
  --input channels.csv \
  --check-channels
```

To comment out inactive rows in place:

```bash
python3 telegram_processor.py \
  --input channels.csv \
  --check-channels \
  --comment-missing
```

`--check-channels` does not require `--start` or `--end`.

Inactive channels are classified from `tdl` output as:

- `not_found`: channel username or target no longer exists
- `inaccessible`: private, forbidden, or otherwise unreachable from the current account
- `error`: an unexpected failure while checking

When `--comment-missing` is used, inactive rows are rewritten like this:

```csv
# SomeChannel,@somechannel,secret # inaccessible
```

## Input CSV Format

The CSV must start with:

- `name`
- `channel`

Supported optional columns:

- `password_source`
- `password`, `password1`, `password2`, and any additional password columns after `channel`

Static-password example:

```csv
name,channel,password
Channel1,@channel1,password123
Channel2,@channel2,
```

Funnel-channel example:

```csv
name,channel,password_source
OnlyLogsCloud,@OnlyLogsCloud,password_in_post
```

### `password_in_post` behavior

- The processor reads the exported Telegram message for each downloaded archive.
- It only accepts passwords found in the same message as the archive file.
- It looks for case-insensitive `pass` or `password` markers followed by a delimiter such as `:`, `=`, `-`, or whitespace.
- Supported examples include:
  - `pass: 123`
  - `Password: @OnlyLogsCloud`
  - `Password FULL LOGS - @BurnCloudLogs`
- Unlabeled promo links and reserve-channel links are ignored.
- If one message contains multiple archives and one matching password, that password is applied to each archive from that message.
- Static password columns are ignored when `password_source=password_in_post`.
- If no same-message password is found, or extraction still fails, that archive is skipped and processing continues.

Already-commented CSV rows are ignored by the channel checker.

## Processing Flow

1. Load channel definitions from the CSV.
2. In normal mode, export channel messages with `tdl chat export` and download matching files.
3. Deduplicate downloaded files across channels with `rdfind`.
4. Extract archives with `7z`.
5. For passworded archives, try either:
   - configured static passwords, then no password
   - or the same-message post password when `password_in_post` is enabled
6. Deduplicate extracted files inside each channel directory.
7. Process stealer-log outputs when archives are present.
8. Combine and sort unique text output into a `*-combo.csv` file.
9. Move final result files into the output directory.
10. Optionally clean channel working directories.

## Output Files

Depending on the channel contents, the script may emit:

- `{channel_name}-{month-year}-combo.csv`
- `{channel_name}-{month-year}-credentials.csv`
- `{channel_name}-{month-year}-autofills.csv`

All final outputs are written to the configured output directory.

## Configuration

`settings.json` controls download, extraction, sorting, and subprocess behavior. The tracked file in this repo is a sample configuration; update the `stealer_log_processor.path` for your machine if you use that integration.

Current sections:

- `stealer_log_processor`
- `tdl`
- `sort`
- `archive`
- `processing`
- `logging`
- `subprocess`

Notable keys:

- `tdl.max_parallel_downloads`
- `tdl.export_channel_threads`
- `tdl.excluded_extensions`
- `archive.extract_patterns`
- `archive.supported_extensions`
- `archive.extract_timeout`
- `archive.max_parallel_extractions`
- `sort.temp_dir`
- `processing.max_workers`

## Notes

- `channels.csv` is intentionally gitignored and treated as local operator data.
- If `tqdm` is not installed, the script still works; it only falls back to plain logging.
