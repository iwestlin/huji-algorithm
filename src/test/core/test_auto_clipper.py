import json
from unittest.mock import patch

from src.main.constant.autoclip_constant import (
    ActionType,
    BadmintonAutoClipConfig,
    MatchType,
)
from src.main.core.auto_clipper import CleanableFileCollection
from src.main.core.badminton_auto_clipper import BadmintonAutoClipper
from src.main.core.pingpong_auto_clipper import PingPongAutoClipper
from src.main.service.large_model_service import LargeModelService
from src.main.utils.path_utils import get_resource
from src.test.base.test_base import TestCaseBase


class TestAutoClipper(TestCaseBase):
    def _setup_internal(self):
        large_model_service = LargeModelService(self.config.large_model_service_config)
        self.ping_pong_auto_clipper: PingPongAutoClipper = PingPongAutoClipper(
            self.config.auto_clip_config.ping_pong,
            self.config.auto_clip_config.common_options,
            large_model_service,
        )
        self.badminton_auto_clipper: BadmintonAutoClipper = BadmintonAutoClipper(
            self.config.auto_clip_config.badminton,
            self.config.auto_clip_config.common_options,
            large_model_service,
        )

    def test_ping_pong_clip(self):
        self.ping_pong_auto_clipper.autoclip_video(get_resource("video/examples/test.mp4"))

    def test_badminton_clip(self):
        self.badminton_auto_clipper.autoclip_video(get_resource("video/examples/blue.mp4"))

    def test_badminton_double_clip(self):
        config = BadmintonAutoClipConfig(match_type=MatchType.DOUBLES_MATCH)
        self.badminton_auto_clipper.autoclip_video(
            input_video_path=get_resource("video/examples/badminton_doubles_1.mp4"),
            auto_clip_config=json.loads(config.model_dump_json()),
        )

    def test_predict_uses_original_video_not_width640_resize(self):
        """Classify 预处理会把短边升到 640;先把整段视频压成宽 640 会再被放大,检出大幅下降。"""
        video_path = "/videos/unique-dir/match.mp4"
        video_info = type("VideoInfo", (), {"video_path": video_path})()
        clipper = self.ping_pong_auto_clipper
        with (
            patch(
                "src.main.utils.video_utils.resize_video_ratio",
            ) as resize,
            patch.object(
                clipper,
                "predict_video_action_points_internal_v2",
                return_value=[],
            ) as predict,
            patch.object(clipper, "_load_cache", return_value=None),
            patch.object(clipper, "_cache_predict_infos", return_value=""),
        ):
            clipper._predict_video_action_points(  # pyright: ignore [reportPrivateUsage]
                CleanableFileCollection(),
                video_info,  # pyright: ignore [reportArgumentType]
                "ping_pong_singles_profession",
                {ActionType.PLAY_BALL.name.lower(): ActionType.PLAY_BALL},
            )
        resize.assert_not_called()
        predict.assert_called_once()
        self.assertEqual(predict.call_args.kwargs["video_path"], video_path)

    def test_prediction_cache_does_not_collide_on_same_stem(self):
        cache_path_for = self.ping_pong_auto_clipper._cache_path_for  # pyright: ignore [reportPrivateUsage]
        path_a = cache_path_for("/videos/a/test.mp4")
        path_b = cache_path_for("/videos/b/test.mp4")
        self.assertNotEqual(path_a, path_b)
        self.assertTrue(path_a.endswith(".json"))
        self.assertTrue(path_b.endswith(".json"))
