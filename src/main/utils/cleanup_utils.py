import contextlib
import time
from pathlib import Path

from src.main.config.config import AutoClipCleanupConfig, CommonAutoClipOptions
from src.main.logger import LOG


class OutputCleaner:
    """剪辑任务结束后按 cleanup 配置清理输出目录。

    目录级清理：删除目录下所有文件及空子目录。
    服务模式（Kafka/HTTP）下默认不清理 clipped（最终合并产物上传后再由
    clean_clipped 控制），CLI 模式下合并视频即最终产物，同样默认保留。
    """

    def __init__(self, common_auto_clip_options: CommonAutoClipOptions) -> None:
        self.cleanup: AutoClipCleanupConfig = common_auto_clip_options.cleanup
        self.clipped_output_dir: Path = Path(common_auto_clip_options.output_dir) / "clipped"
        self.resized_output_dir: Path = Path(common_auto_clip_options.output_dir) / "resized"
        self.cache_path: Path = Path(common_auto_clip_options.cache_path)
        self.debug_clip_frame_output_dir: str | None = (
            common_auto_clip_options.debug_clip_frame_output_dir
        )

    def clean_after_clip(self) -> None:
        """剪辑完成后清理临时目录（含过期推理缓存）。"""
        if not self.cleanup.enabled:
            return
        if self.cleanup.clean_resized:
            self._clean_dir(self.resized_output_dir, "resized")
        if self.cleanup.clean_clipped:
            self._clean_dir(self.clipped_output_dir, "clipped")
        if self.cleanup.clean_snippet:
            self._clean_dir(self.debug_clip_frame_output_dir, "clip_snippet")
        self._clean_expired_cache()

    def _clean_dir(self, directory: str | Path | None, name: str) -> None:
        if not directory:
            return
        dir_path: Path = Path(directory)
        if not dir_path.exists():
            return

        removed_count: int = 0
        freed_bytes: int = 0
        for file_path in dir_path.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                freed_bytes += file_path.stat().st_size
                file_path.unlink()
                removed_count += 1
            except OSError as e:
                LOG.warning(f"清理文件失败: {file_path}, {e}")
        # 删除清理后遗留的空子目录
        for sub_dir in sorted((p for p in dir_path.rglob("*") if p.is_dir()), reverse=True):
            with contextlib.suppress(OSError):
                sub_dir.rmdir()

        if removed_count > 0:
            LOG.info(
                f"清理{name}目录完成: {dir_path}, "
                f"删除 {removed_count} 个文件, 释放 {freed_bytes / 1024 / 1024:.1f} MB"
            )

    def _clean_expired_cache(self) -> None:
        retention_days: int = self.cleanup.cache_retention_days
        if retention_days <= 0:
            return
        if not self.cache_path.exists():
            return

        expire_seconds: float = retention_days * 24 * 3600
        now: float = time.time()
        removed_count: int = 0
        for cache_file in self.cache_path.iterdir():
            if not cache_file.is_file():
                continue
            try:
                if now - cache_file.stat().st_mtime > expire_seconds:
                    cache_file.unlink()
                    removed_count += 1
            except OSError as e:
                LOG.warning(f"清理推理缓存失败: {cache_file}, {e}")
        if removed_count > 0:
            LOG.info(f"清理过期推理缓存完成: 删除 {removed_count} 个文件")
