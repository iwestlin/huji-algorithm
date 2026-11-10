import bisect
import json
import math
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from src.main.constant.autoclip_constant import VideoBaseInfo, VideoInfo
from src.main.logger import LOG
from src.main.utils import path_utils


ffmpeg_path: str = "ffmpeg"
ffprobe_path: str = "ffprobe"
log_level: str = "error"

_keyframe_cache: dict[str, tuple[float, int, list[float]]] = {}


def merge_videos_by_ffmpeg(
    input_files: list[str],
    output_file: str,
    codec: str,
    bit_rate: str,
    copy_video: bool = False,
) -> None:
    """
    使用 FFmpeg 合并多个视频文件，支持 GPU 加速

    参数:
    - input_files: 输入视频文件列表
    - output_file: 输出视频文件名
    - codec: 视频编码格式
    - audio_codec: 音频编码格式
    """
    for file in input_files:
        if not Path(file).exists():
            raise FileNotFoundError(f"文件不存在: {file}")

    with TemporaryDirectory() as temp_dir:
        # 创建文件列表
        temp_filelist = path_utils.path_join(temp_dir, "filelist.txt")
        with Path(temp_filelist).open(mode="w", encoding="utf-8") as f:
            for file in input_files:
                escaped_file = file.replace("'", "'\\''")
                f.write(f"file '{escaped_file}'\n")

        # 合并视频
        cmd: list[str] = [
            ffmpeg_path,
            "-loglevel",
            log_level,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            temp_filelist,
        ]

        if copy_video:
            # 片段本身就是无损 -c copy 裁出来的且编码参数一致,可直接拼接,避免二次编码
            cmd.extend(["-c", "copy"])
        else:
            cmd.extend(
                [
                    "-preset",
                    "fast",
                    "-c:v",
                    get_acc_codec(codec),
                    "-b:v",
                    bit_rate,
                ]
            )

        cmd.append(output_file)
        _run_cmd(cmd)


def _run_cmd(cmd: list[str]) -> str:
    """执行命令"""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        LOG.error(f"命令执行失败: cmd={cmd}, stderr={e.stderr}")
        raise e


def get_video_base_info(input_file: str) -> VideoBaseInfo:
    cmd: list[str] = [
        "ffprobe",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        input_file,
    ]

    result = _run_cmd(cmd)

    info = json.loads(result).get("format", [])
    if not info:
        raise ValueError("未找到视频流")
    return VideoBaseInfo(duration=float(info.get("duration", 0)), size=int(info.get("size", 0)))


def get_video_info(input_file: str) -> "VideoInfo":
    """获取视频的完整信息，包括帧率、时长、总帧数和是否为可变帧率"""
    cmd: list[str] = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,duration,nb_frames,codec_name,bit_rate",
        "-of",
        "json",
        input_file,
    ]

    result = _run_cmd(cmd)

    streams = json.loads(result).get("streams", [])

    if not streams:
        raise ValueError("未找到视频流")

    stream = streams[0]

    # 解析帧率的辅助函数
    def parse_frame_rate(rate_str: str) -> float:
        if "/" in rate_str:
            num, den = map(int, rate_str.split("/"))
            return num / den
        return float(rate_str)

    codec_name: str = stream.get("codec_name", "")

    # 获取帧率信息
    r_frame_rate_str: str = stream.get("r_frame_rate", "0/1")
    avg_frame_rate_str: str = stream.get("avg_frame_rate", "0/1")

    r_frame_rate: float = parse_frame_rate(r_frame_rate_str)
    avg_frame_rate: float = parse_frame_rate(avg_frame_rate_str)

    # 使用平均帧率作为主要帧率
    fps: float = avg_frame_rate if avg_frame_rate > 0 else r_frame_rate

    # 获取时长和总帧数
    duration: float = float(stream.get("duration", 0))
    total_frames: int = int(stream.get("nb_frames", 0))
    bit_rate: str = stream.get("bit_rate")

    # 如果无法获取总帧数，通过时长计算
    if total_frames == 0 and fps > 0 and duration > 0:
        total_frames = int(duration * fps)

    # 判断是否为可变帧率 (VFR)
    # 当r_frame_rate和avg_frame_rate差异明显时认为是VFR
    is_vfr: bool = (
        abs(r_frame_rate - avg_frame_rate) > 0 if r_frame_rate > 0 and avg_frame_rate > 0 else False
    )

    return VideoInfo(
        fps=fps,
        duration=duration,
        total_frames=total_frames,
        is_vfr=is_vfr,
        r_frame_rate_str=r_frame_rate_str,
        avg_frame_rate_str=avg_frame_rate_str,
        r_frame_rate_val=r_frame_rate,
        avg_frame_rate_val=avg_frame_rate,
        video_path=input_file,
        video_file=Path(input_file),
        codec_name=codec_name,
        bit_rate=bit_rate,
    )


def convert_to_cfr_if_variable_frame_rate(input_file: str, output_file: str) -> None:
    """将视频转换为恒定帧率(CFR)

    参数:
        input_file: 输入视频文件路径
        output_file: 输出视频文件路径
        codec: 编码器，默认为h264_nvenc(NVIDIA GPU加速)
        crf: 恒定速率因子，取值范围0-51，默认23
        fps: 目标帧率，默认为输入视频的平均帧率
    """
    # 分析视频获取帧率信息
    video_info = get_video_info(input_file)
    if not video_info.is_vfr:
        return

    LOG.info(f"视频 {input_file} 为可变帧率，将转换为恒定帧率")
    print(f"开始转换: {input_file} -> {output_file}")
    print(
        f"使用编码器: {video_info.codec_name}, 恒定帧率: {math.ceil(video_info.r_frame_rate_val)}"
    )
    cmd = [
        ffmpeg_path,
        "-loglevel",
        "error",
        "-i",
        input_file,
        "-vf",
        f"fps={math.ceil(video_info.avg_frame_rate_val)}",
        "-r",
        str(math.ceil(video_info.avg_frame_rate_val)),
        "-c:v",
        get_acc_codec(video_info.codec_name),
        "-vsync",
        "cfr",
        "-preset",
        "fast",
        "-b:v",
        video_info.bit_rate,
        "-movflags",
        "+faststart",
        "-y",
        output_file,
    ]

    _run_cmd(cmd)
    LOG.info(f"转换完成, 输出文件大小: {Path(output_file).stat().st_size / (1024 * 1024):.2f} MB")
    return


def _detect_hardware_acceleration():
    """自动检测可用的硬件加速方式"""
    # 检查 NVIDIA CUDA
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-encoders",
    ]

    result = _run_cmd(cmd)
    if "h264_nvenc" in result or "hevc_nvenc" in result:
        return "cuda"

    # 检查 AMD AMF
    if "h264_amf" in result or "hevc_amf" in result:
        return "amf"

    # 检查 Intel QSV
    if "h264_qsv" in result or "hevc_qsv" in result:
        return "qsv"

    return "none"


def get_acc_codec(codec: str):
    gpu = _detect_hardware_acceleration()

    if gpu.lower() == "cuda":
        return codec + "_nvenc"
    if gpu.lower() == "amf":
        return codec + "_" + "amf"
    if gpu.lower() == "qsv":
        return codec + "_" + "qsv"
    return codec


def clip_video_by_frames(
    video_info: VideoInfo, input_file: str, output_file: str, start_frame: int, end_frame: int
) -> None:
    """根据帧数裁剪视频

    参数:
        video_info: 视频信息
        input_file: 输入视频文件路径
        output_file: 输出视频文件路径
        start_frame: 起始帧
        end_frame: 结束帧
    """
    if start_frame < 0 or end_frame > video_info.total_frames:
        raise ValueError(f"帧数超出范围: 0-{video_info.total_frames}")
    if start_frame >= end_frame:
        raise ValueError("起始帧必须小于结束帧")

    start_time: float = start_frame / video_info.fps
    duration: float = (end_frame - start_frame) / video_info.fps
    return clip_video_by_times(input_file, start_time, duration, output_file)


def _get_video_keyframes(input_file: str) -> list[float]:
    """获取视频所有关键帧时间点(按输入文件缓存,避免每个片段都扫一遍)。"""
    abs_path = os.path.abspath(input_file)
    try:
        stat = os.stat(abs_path)
    except OSError:
        return []
    cached = _keyframe_cache.get(abs_path)
    if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]

    cmd: list[str] = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "csv=p=0",
        abs_path,
    ]
    try:
        result = _run_cmd(cmd)
    except subprocess.CalledProcessError as e:
        LOG.warning(f"读取关键帧失败,将按原时间裁剪: {e}")
        return []

    keyframes: list[float] = []
    for line in result.splitlines():
        try:
            keyframes.append(float(line.strip().rstrip(",")))
        except ValueError:
            continue
    keyframes.sort()
    _keyframe_cache[abs_path] = (stat.st_mtime, stat.st_size, keyframes)
    return keyframes


def _get_keyframe_before(input_file: str, seconds: float) -> float | None:
    keyframes = _get_video_keyframes(input_file)
    if not keyframes:
        return None
    index = bisect.bisect_right(keyframes, seconds) - 1
    return keyframes[index] if index >= 0 else None


def _get_keyframe_after(input_file: str, seconds: float) -> float | None:
    keyframes = _get_video_keyframes(input_file)
    if not keyframes:
        return None
    index = bisect.bisect_left(keyframes, seconds)
    return keyframes[index] if index < len(keyframes) else None


def clip_video_by_times(
    input_file: str,
    start_time: float,
    duration: float,
    output_file: str,
    align_to_keyframe: bool = False,
    min_start_time: float | None = None,
) -> None:
    """根据时间裁剪视频

    align_to_keyframe=True 时(仅用于 -c copy 拼接):起点必须落在关键帧上,
    否则 ffmpeg 会保留“关键帧到起点”之间带负 PTS/discard 标记的帧。这些帧
    被拼接时会被压到上一段末尾,造成接缝处画面冻结/闪烁。因此这里把起点前移到
    最近的关键帧,并延长到原定终点。

    min_start_time 用于避免与上一段重叠:若起点的关键帧早于上一段结束时间,
    则改用上一段结束之后的第一个关键帧。
    """
    end_time = start_time + duration
    keyframe: float | None = None
    if align_to_keyframe:
        keyframe = _get_keyframe_before(input_file, start_time)
        if (
            keyframe is not None
            and min_start_time is not None
            and keyframe < min_start_time
        ):
            keyframe = _get_keyframe_after(input_file, min_start_time)
            if keyframe is None or keyframe >= end_time:
                keyframe = None
    actual_start = keyframe if keyframe is not None else start_time
    actual_duration = max(end_time - actual_start, 0.001)
    if actual_start != start_time:
        LOG.debug(
            f"裁剪起点前移到关键帧: {start_time:.3f} -> {actual_start:.3f},"
            f" 时长 {duration:.3f} -> {actual_duration:.3f}"
        )

    # 构建 FFmpeg 命令
    cmd: list[str] = [
        ffmpeg_path,
        "-loglevel",
        log_level,
        "-y",
        "-ss",
        str(actual_start),  # 先定位到关键帧
        "-i",
        input_file,  # 然后读取输入文件
        "-t",
        str(actual_duration),  # 设置持续时间
        "-c",
        "copy",
        output_file,
    ]

    _run_cmd(cmd)


def convert_to_editable_format(
    input_file: str,
    output_file: str,
    codec: str = "h264",
) -> None:
    """将视频转换为易于编辑的格式"""
    if not Path(input_file).exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    video_info = get_video_info(input_file)

    cmd: list[str] = [
        ffmpeg_path,
        "-loglevel",
        log_level,
        "-y",
        "-i",
        input_file,
        "-c:v",
        get_acc_codec(codec),
        "-vsync",
        "cfr",
        "-r",
        str(video_info.fps),
        output_file,
    ]

    _run_cmd(cmd)


def resize_video_ratio(input_file: str, output_file: str, width: int):
    video_info: VideoInfo = get_video_info(input_file)

    """按宽度缩放，自动保持宽高比"""
    cmd: list[str] = [
        "ffmpeg",
        "-loglevel",
        log_level,
        "-i",
        input_file,
        "-vf",
        f"scale={width}:-1",  # -1表示自动计算高度保持比例
        "-c:v",
        get_acc_codec(video_info.codec_name),
        "-y",
        output_file,
    ]
    _run_cmd(cmd)


def interval_extract_frames(video_path: str, frame_interval: int, temp_dir: str) -> None:
    """
    按指定的帧间隔提取视频帧
    参数:
        video_path: 视频文件路径
        frame_interval: 帧间隔，每秒提取多少帧
        output_dir: 输出图像目录
    """
    cmd: list[str] = [
        ffmpeg_path,
        "-loglevel",
        log_level,
        "-i",
        video_path,
        "-vf",
        # 短边缩放到 classify imgsz,长边按比例。随后 preprocess_image 只做 CenterCrop,
        # 避免原分辨率 JPEG + PIL 二次放大。flags=bilinear 贴近 ultralytics Resize。
        (
            f"fps={frame_interval},"
            "scale='if(lt(iw,ih),640,-1)':'if(lt(iw,ih),-1,640)':flags=bilinear"
        ),
        "-q:v",
        "3",
        "-y",
        str(Path(temp_dir, "%d.jpg")),
    ]
    _run_cmd(cmd)
