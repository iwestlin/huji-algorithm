#!/usr/bin/env bash
set -euo pipefail

usage() {
    local ping_default="" badminton_default=""
    if [[ "$default_sport" == "ping_pong" ]]; then
        ping_default="（默认）"
    else
        badminton_default="（默认）"
    fi
    cat <<EOF
用法: ${invoked_name} [选项] <视频文件> [main.py 参数...]

快捷参数:
  -d, --doubles       羽毛球双打
  -s, --singles       羽毛球单打
  -p, --ping-pong     乒乓球${ping_default}
  -b, --badminton     羽毛球${badminton_default}
  -h, --help          显示帮助

其余参数（-o/--output-dir、--cleanup、--no-cleanup、--set 等）原样透传。

示例:
  ${invoked_name} input.mp4
  ${invoked_name} -d input.mp4
  ${invoked_name} -o ~/clips input.mp4
EOF
}

invoked_name="$(basename "$0")"

source_path="${BASH_SOURCE[0]}"
while [ -h "$source_path" ]; do
    dir="$(cd -P "$(dirname "$source_path")" >/dev/null 2>&1 && pwd)"
    source_path="$(readlink "$source_path")"
    [[ "$source_path" != /* ]] && source_path="$dir/$source_path"
done
script_dir="$(cd -P "$(dirname "$source_path")" >/dev/null 2>&1 && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
python_bin="$project_root/.venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
    echo "未找到虚拟环境: $python_bin" >&2
    echo "请先运行: cd \"$project_root\" && ./setup.sh" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "未找到 ffmpeg，请先安装并确保在 PATH 中" >&2
    exit 1
fi

default_sport="badminton"
case "$invoked_name" in
    *ppcut*) default_sport="ping_pong" ;;
esac

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 1
fi

forward=()
video_set=0
sport_set=0
match_set=0
seen_positional=0

needs_value=(
    -o --output-dir -v --video-path --config
    --sport --match-type --set
)

while [[ $# -gt 0 ]]; do
    arg="$1"
    case "$arg" in
        -h|--help)
            usage
            exit 0
            ;;
        -p|--ping-pong)
            forward+=(--sport ping_pong)
            sport_set=1
            shift
            ;;
        -b|--badminton)
            forward+=(--sport badminton)
            sport_set=1
            shift
            ;;
        -d|--doubles)
            forward+=(--match-type doubles)
            match_set=1
            shift
            ;;
        -s|--singles)
            forward+=(--match-type singles)
            match_set=1
            shift
            ;;
        -v|--video-path)
            if [[ $# -lt 2 ]]; then
                echo "$arg 需要一个视频路径" >&2
                exit 1
            fi
            forward+=(--video-path "$2")
            video_set=1
            shift 2
            ;;
        --video-path=*)
            forward+=(--video-path "${arg#*=}")
            video_set=1
            shift
            ;;
        --sport=*|--match-type=*|--config=*|--output-dir=*|--set=*)
            forward+=("$arg")
            [[ "$arg" == --sport=* ]] && sport_set=1
            [[ "$arg" == --match-type=* ]] && match_set=1
            [[ "$arg" == --video-path=* ]] && video_set=1
            shift
            ;;
        -*)
            forward+=("$arg")
            for key in "${needs_value[@]}"; do
                if [[ "$arg" == "$key" ]]; then
                    if [[ $# -lt 2 ]]; then
                        echo "$arg 需要一个参数值" >&2
                        exit 1
                    fi
                    forward+=("$2")
                    [[ "$key" == "--sport" ]] && sport_set=1
                    [[ "$key" == "--match-type" ]] && match_set=1
                    [[ "$key" == "--video-path" || "$key" == "-v" ]] && video_set=1
                    shift
                    break
                fi
            done
            shift
            ;;
        *)
            if [[ $video_set -eq 0 && $seen_positional -eq 0 ]]; then
                forward+=(--video-path "$arg")
                video_set=1
            else
                forward+=("$arg")
            fi
            seen_positional=1
            shift
            ;;
    esac
done

if [[ $video_set -eq 0 ]]; then
    echo "缺少视频文件路径" >&2
    usage >&2
    exit 1
fi

run_dir="$PWD"

defaults=(--sport "$default_sport")
if [[ "$default_sport" == "badminton" && $match_set -eq 0 ]]; then
    defaults+=(--match-type singles)
fi
defaults+=(
    --set "autoclip_service_config.common.output_dir=$run_dir/output/clipped"
    --set "autoclip_service_config.common.cache_path=$run_dir/output/cache"
    --set "autoclip_service_config.common.debug_clip_frame_output_dir=$run_dir/output/clip_snippet"
)

exec "$python_bin" "$project_root/main.py" "${defaults[@]}" "${forward[@]}"
