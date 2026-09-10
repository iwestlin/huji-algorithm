# Huji Algorithm

乒乓球、羽毛球比赛视频自动剪辑：检测比赛片段，过滤捡球/回放，输出精彩片段。

[English](README.en.md)

## 功能

- 传入视频文件，自动检测并剪辑比赛过程
- 支持乒乓球、羽毛球（单打/双打）
- 支持本地文件与对象存储（COS/OSS）远程处理

## 前置依赖

- Python 3.12+
- [FFmpeg](https://ffmpeg.org/)（需在 PATH 中，剪辑合并视频时使用）
- GPU 可选（ncnn 推理支持 Vulkan GPU 加速，无 GPU 时自动回退 CPU）

## 快速开始

### 本地快速剪辑（推荐）

**Linux / macOS**

```bash
./setup.sh
source .venv/bin/activate

python main.py --video-path videos/demo.mp4 --sport ping_pong
python main.py -v src/resources/video/examples/test.mp4 --sport badminton --match-type doubles
```

**Windows（PowerShell）**

```powershell
.\setup.ps1
.venv\Scripts\activate

python main.py --video-path videos\demo.mp4 --sport ping_pong
python main.py -v src\resources\video\examples\test.mp4 --sport badminton --match-type doubles
```

若无法执行 `setup.ps1`（执行策略限制），可手动安装：

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py --video-path videos\demo.mp4 --sport ping_pong
```

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--video-path` / `-v` | 本地视频文件路径 |
| `--sport` | 运动类型：`ping_pong` 或 `badminton`（clip 模式必填） |
| `--match-type` | 羽毛球比赛类型：`singles`（默认）/ `doubles` |
| `--output-dir` / `-o` | 剪辑输出目录（覆盖配置文件中的 `output_dir`） |
| `--cleanup` | 剪辑完成后清理临时输出（覆盖 cleanup 配置的 `enabled`） |
| `--no-cleanup` | 剪辑完成后不清理临时输出（覆盖 cleanup 配置的 `enabled`） |
| `--config` | 配置目录，默认 `src/resources` |

剪辑完成后，终端会打印合并视频的输出路径；默认写入配置中的 `output/clipped/`（可用 `-o` 覆盖）。

除上表外，所有配置文件参数均可通过命令行覆盖（`--set 点号路径=值`，可重复），完整参数说明见 `src/main/config/config.py` 与 `src/resources/application*.yml`。

示例视频：`src/resources/video/examples/`。

仓库内 `src/resources/application*.yml` 已包含可运行的默认配置（本地 Kafka、示例模型路径）。  
使用远程 COS 或对接后台 API 时，只需修改其中的占位项：

| 配置项 | 说明 |
|--------|------|
| `filesystem.cos.*` | 对象存储密钥与 bucket |
| `internal.http.access_token` | 后台 API 令牌 |
| `datasource.mysql.*` | MySQL（远程文件配置查询） |

也可用环境变量覆盖（点号改为下划线），见 `.env.example`。支持 Jasypt：`ENC(密文)` + `JASYPT_PASSWORD`。

### 输出目录与自动清理

剪辑过程中的文件默认写在 `output/` 下：

| 目录 | 内容 |
|------|------|
| `output/clipped/clipped/` | 最终合并视频（结果产物） |
| `output/clipped/resized/` | 预测用的缩小视频（中间产物） |
| `output/cache/` | 推理结果缓存（按视频文件名） |
| `output/clip_snippet/` | 调试帧输出（`debug_clip_frame_output_dir` 开启时） |
| `output/logs/` | 运行日志（50 MB 轮转，保留 30 天） |

剪辑结束后会按 `autoclip_service_config.common.cleanup` 配置自动清理：

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `enabled` | `False` | 清理总开关 |
| `clean_resized` | `True` | 清理预测用缩小视频（中间产物） |
| `clean_clipped` | `False` | 清理最终合并视频目录（一般无需开启） |
| `clean_snippet` | `False` | 清理调试帧输出目录 |
| `cache_retention_days` | `7` | 推理缓存保留天数，`0` 表示永不过期 |

缓存按视频文件名命中，同名视频复用上次推理结果；修改模型或参数后如需强制重算，可删除对应缓存文件或调低保留天数。

## 如何开发

### 安装依赖

Linux / macOS：`./setup.sh`  
Windows：`.\setup.ps1`

### 环境配置

- `application.yml`：公共配置
- `application_dev.yml`：`env: dev` 时加载
- `application_prod.yml`：`env: prod` 时加载

### 测试

```bash
python -m unittest discover -s src/test -p "test_*.py"
```

## 模型训练

训练与模型导出已分离到 [huji-train](https://github.com/hhoao/huji-train)。
本仓库只做推理（ncnn），模型位于 `src/resources/ncnn_models/`。

## 社区

- **QQ 群**：[112856301](https://qm.qq.com/q/U5U0zLyyA2)（点击加入）
- **Discord**：[加入群组](https://discord.com/channels/1518551459053178960/1518551461242474558)
