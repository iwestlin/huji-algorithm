import tempfile
import time
from pathlib import Path
from unittest import TestCase

from src.main.config.config import AutoClipCleanupConfig, CommonAutoClipOptions
from src.main.utils.cleanup_utils import OutputCleaner


class _StubCleanup(AutoClipCleanupConfig):
    """绕过 yml 配置，直接构造 cleanup 选项。"""

    def __init__(
        self,
        enabled: bool = True,
        clean_resized: bool = True,
        clean_clipped: bool = False,
        clean_snippet: bool = False,
        cache_retention_days: int = 7,
    ):
        self.enabled = enabled
        self.clean_resized = clean_resized
        self.clean_clipped = clean_clipped
        self.clean_snippet = clean_snippet
        self.cache_retention_days = cache_retention_days


class _StubCommonOptions(CommonAutoClipOptions):
    def __init__(self, root: str, cleanup: AutoClipCleanupConfig):
        self.output_dir = str(Path(root, "output", "clipped"))
        self.cache_path = str(Path(root, "output", "cache"))
        self.debug_clip_frame_output_dir = str(Path(root, "output", "clip_snippet"))
        self.cleanup = cleanup


class TestOutputCleaner(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.clipped_dir = Path(self.root, "output", "clipped", "clipped")
        self.resized_dir = Path(self.root, "output", "clipped", "resized")
        self.cache_dir = Path(self.root, "output", "cache")
        self.snippet_dir = Path(self.root, "output", "clip_snippet", "video1")
        for d in (self.clipped_dir, self.resized_dir, self.cache_dir, self.snippet_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.merged_video = self.clipped_dir / "merged.mp4"
        self.merged_video.write_bytes(b"x" * 1024)
        (self.resized_dir / "video.mp4").write_bytes(b"y" * 2048)
        (self.snippet_dir / "frame.jpg").write_bytes(b"z")
        (self.cache_dir / "video.json").write_text("{}")

    def tearDown(self):
        self._tmp.cleanup()

    def _make_cleaner(
        self,
        *,
        enabled: bool = True,
        clean_resized: bool = True,
        clean_clipped: bool = False,
        clean_snippet: bool = False,
        cache_retention_days: int = 7,
    ) -> OutputCleaner:
        return OutputCleaner(
            _StubCommonOptions(
                self.root,
                _StubCleanup(
                    enabled=enabled,
                    clean_resized=clean_resized,
                    clean_clipped=clean_clipped,
                    clean_snippet=clean_snippet,
                    cache_retention_days=cache_retention_days,
                ),
            )
        )

    def test_disabled_cleans_nothing(self):
        self._make_cleaner(enabled=False).clean_after_clip()
        self.assertTrue(self.merged_video.exists())
        self.assertTrue((self.resized_dir / "video.mp4").exists())

    def test_clean_resized_only_keeps_clipped_and_snippet(self):
        self._make_cleaner().clean_after_clip()
        self.assertFalse((self.resized_dir / "video.mp4").exists())
        self.assertTrue(self.merged_video.exists())
        self.assertTrue((self.snippet_dir / "frame.jpg").exists())
        self.assertTrue((self.cache_dir / "video.json").exists())

    def test_clean_clipped_removes_merged_output(self):
        self._make_cleaner(clean_clipped=True).clean_after_clip()
        self.assertFalse(self.merged_video.exists())
        # 空子目录应一并移除
        self.assertFalse(self.clipped_dir.exists() and any(self.clipped_dir.iterdir()))

    def test_clean_snippet(self):
        self._make_cleaner(clean_snippet=True).clean_after_clip()
        self.assertFalse((self.snippet_dir / "frame.jpg").exists())

    def test_cache_ttl_expired_and_fresh(self):
        expired = self.cache_dir / "old.json"
        expired.write_text("{}")
        past = time.time() - 8 * 24 * 3600
        import os

        os.utime(expired, (past, past))
        self._make_cleaner().clean_after_clip()
        self.assertFalse(expired.exists())
        self.assertTrue((self.cache_dir / "video.json").exists())

    def test_cache_ttl_zero_disables(self):
        (self.cache_dir / "video.json").write_text("{}")
        past = time.time() - 365 * 24 * 3600
        import os

        os.utime(self.cache_dir / "video.json", (past, past))
        self._make_cleaner(cache_retention_days=0).clean_after_clip()
        self.assertTrue((self.cache_dir / "video.json").exists())
