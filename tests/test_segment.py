"""segment 유닛 테스트 – Segment 데이터클래스와 split_video."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.engine.segment import Segment, split_video


class TestSegment:
    def test_defaults(self):
        seg = Segment(index=0, start=0.0, end=30.0)
        assert seg.video_description == ""
        assert seg.transcript == ""
        assert seg.transcript_segments == []
        assert seg.score == 0.0
        assert seg.is_highlight is False
        assert seg.reason == ""
        assert seg.tags == []
        assert seg.suggested_title == ""

    def test_duration_property(self):
        seg = Segment(index=0, start=10.0, end=40.0)
        assert seg.duration == pytest.approx(30.0)

    def test_frame_timestamps_default(self):
        seg = Segment(index=0, start=0.0, end=30.0)
        assert seg.frame_timestamps == []


class TestSplitVideo:
    @patch("src.engine.segment.get_video_duration", return_value=90.0)
    @patch("src.engine.segment.sample_timestamps")
    def test_splits_into_correct_count(self, mock_sample, mock_dur):
        mock_sample.side_effect = lambda s, e, n: [s]
        segments = split_video(Path("/fake/video.mp4"), segment_length=30, frames_per_segment=6)
        assert len(segments) == 3
        assert segments[0].start == 0.0
        assert segments[0].end == 30.0
        assert segments[1].start == 30.0
        assert segments[2].end == 90.0

    @patch("src.engine.segment.get_video_duration", return_value=50.0)
    @patch("src.engine.segment.sample_timestamps")
    def test_last_segment_clipped(self, mock_sample, mock_dur):
        mock_sample.side_effect = lambda s, e, n: [s]
        segments = split_video(Path("/fake/video.mp4"), segment_length=30, frames_per_segment=6)
        assert len(segments) == 2
        assert segments[1].end == 50.0  # 30~50초 (not 30~60)

    @patch("src.engine.segment.get_video_duration", return_value=10.0)
    @patch("src.engine.segment.sample_timestamps")
    def test_short_video_single_segment(self, mock_sample, mock_dur):
        mock_sample.side_effect = lambda s, e, n: [s]
        segments = split_video(Path("/fake/video.mp4"), segment_length=30, frames_per_segment=6)
        assert len(segments) == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 10.0

    @patch("src.engine.segment.get_video_duration", return_value=60.0)
    @patch("src.engine.segment.sample_timestamps", return_value=[0.0, 5.0, 10.0])
    def test_frame_timestamps_assigned(self, mock_sample, mock_dur):
        segments = split_video(Path("/fake/video.mp4"), segment_length=30, frames_per_segment=3)
        assert segments[0].frame_timestamps == [0.0, 5.0, 10.0]
