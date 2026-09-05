# Huji Algorithm

Auto-clip table tennis and badminton match videos: detect rally segments, filter ball pickup/replay, output highlights.

[中文](README.md)

## Features

- Pass in a video file; auto-detect and clip match play
- Table tennis and badminton (singles / doubles)
- Local files and remote object storage (COS/OSS)

## Prerequisites

- Python 3.12+
- [FFmpeg](https://ffmpeg.org/) on `PATH` (used when merging clipped segments)
- GPU optional; on Windows if PyTorch install fails, pick the right CUDA build from [pytorch.org](https://pytorch.org) and `pip install`

## Quick start

### Local quick clip (recommended)

**Linux / macOS**

```bash
./setup.sh
source .venv/bin/activate

python main.py --video-path videos/demo.mp4 --sport ping_pong
python main.py -v src/resources/video/examples/test.mp4 --sport badminton --match-type doubles
```

**Windows (PowerShell)**

```powershell
.\setup.ps1
.venv\Scripts\activate

python main.py --video-path videos\demo.mp4 --sport ping_pong
python main.py -v src\resources\video\examples\test.mp4 --sport badminton --match-type doubles
```

If `setup.ps1` is blocked by execution policy, install manually:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py --video-path videos\demo.mp4 --sport ping_pong
```

### CLI flags

| Flag | Description |
| ---- | ----------- |
| `--video-path` / `-v` | Local video file path |
| `--sport` | `ping_pong` or `badminton` (required for clip mode) |
| `--match-type` | Badminton: `singles` (default) / `doubles` |
| `--output-dir` / `-o` | Clip output dir (overrides `output_dir` in config) |
| `--cleanup` | Clean temp output after clipping (overrides cleanup `enabled`) |
| `--no-cleanup` | Keep temp output after clipping (overrides cleanup `enabled`) |
| `--config` | Config directory, default `src/resources` |
| `--train` | Train models |

When clipping finishes, the merged video path is printed to the terminal; default output is `output/clipped/` (override with `-o`).

Beyond the table above, every config-file key can be overridden on the command line (`--set dotted.path=value`, repeatable). For the full list of keys see `src/main/config/config.py` and `src/resources/application*.yml`.

Sample videos: `src/resources/video/examples/`.

`src/resources/application*.yml` ships with runnable defaults (local Kafka, sample model paths).  
For remote COS or backend API integration, update placeholders:

| Config key | Description |
| ---------- | ----------- |
| `filesystem.cos.*` | Object storage keys & bucket |
| `internal.http.access_token` | Backend API token |
| `datasource.mysql.*` | MySQL (remote file config lookup) |

Environment variables can override config (dots → underscores); see `.env.example`. Jasypt supported: `ENC(ciphertext)` + `JASYPT_PASSWORD`.

### Output directories & auto cleanup

Files produced during clipping default to `output/`:

| Directory | Content |
| --------- | ------- |
| `output/clipped/clipped/` | Final merged videos (result artifacts) |
| `output/clipped/resized/` | Downscaled videos for prediction (intermediate) |
| `output/cache/` | Inference result cache (per video file name) |
| `output/clip_snippet/` | Debug frame output (when `debug_clip_frame_output_dir` is set) |
| `output/logs/` | Logs (50 MB rotation, 30-day retention) |

After each clip finishes, directories are cleaned automatically per `autoclip_service_config.common.cleanup`:

| Config key | Default | Description |
| ---------- | ------- | ----------- |
| `enabled` | `False` | Master switch for cleanup |
| `clean_resized` | `True` | Remove downscaled prediction videos (intermediate) |
| `clean_clipped` | `False` | Remove final merged video directory (rarely needed) |
| `clean_snippet` | `False` | Remove debug frame output directory |
| `cache_retention_days` | `7` | Inference cache retention in days; `0` disables expiry |

The cache is keyed by video file name, so the same video reuses its previous inference results. To force recomputation after changing models or parameters, delete the matching cache file or lower the retention.

## Development

### Install dependencies

Linux / macOS: `./setup.sh`  
Windows: `.\setup.ps1`

### Configuration

- `application.yml` — shared config
- `application_dev.yml` — loaded when `env: dev`
- `application_prod.yml` — loaded when `env: prod`

### Tests

```bash
python -m unittest discover -s src/test -p "test_*.py"
```

## Community

- **QQ Group**: [112856301](https://qm.qq.com/q/U5U0zLyyA2) (click to join)
- **Discord**: [Join our server](https://discord.com/channels/1518551459053178960/1518551461242474558)
