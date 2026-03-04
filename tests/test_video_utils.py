"""video_utils 유닛 테스트 – FFmpeg/FFprobe 호출을 mock 처리."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.utils.video_utils import (
    extract_audio,
    extract_frames,
    get_video_duration,
    load_frames_as_pil,
    sample_timestamps,
)


class TestGetVideoDuration:
    @patch("src.utils.video_utils.subprocess.run")
    def test_returns_float(self, mock_run):
        mock_run.return_value = MagicMock(stdout="120.5\n", returncode=0)
        duration = get_video_duration(Path("/fake/video.mp4"))
        assert duration == pytest.approx(120.5)

    @patch("src.utils.video_utils.subprocess.run")
    def test_calls_ffprobe(self, mock_run):
        mock_run.return_value = MagicMock(stdout="60.0\n", returncode=0)
        get_video_duration(Path("/fake/video.mp4"))
        args = mock_run.call_args[0][0]
        assert args[0] == "ffprobe"


class TestSampleTimestamps:
    def test_zero_frames(self):
        assert sample_timestamps(0.0, 30.0, 0) == []

    def test_single_frame_is_midpoint(self):
        result = sample_timestamps(0.0, 30.0, 1)
        assert len(result) == 1
        assert result[0] == pytest.approx(15.0)

    def test_two_frames(self):
        result = sample_timestamps(0.0, 30.0, 2)
        assert len(result) == 2
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(30.0)

    def test_six_frames_uniform(self):
        result = sample_timestamps(0.0, 30.0, 6)
        assert len(result) == 6
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(30.0)

    def test_offset_range(self):
        result = sample_timestamps(60.0, 90.0, 4)
        assert result[0] == pytest.approx(60.0)
        assert result[-1] == pytest.approx(90.0)


class TestExtractFrames:
    @patch("src.utils.video_utils.subprocess.run")
    def test_creates_output_files(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        result = extract_frames(
            Path("/fake/video.mp4"), [0.0, 5.0, 10.0], tmp_path / "frames"
        )
        assert len(result) == 3
        assert all(str(p).endswith(".png") for p in result)
        assert mock_run.call_count == 3


class TestLoadFramesAsPil:
    def test_loads_images(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"frame_{i}.png"
            Image.new("RGB", (64, 64), color=(i * 80, 0, 0)).save(str(p))
            paths.append(p)
        frames = load_frames_as_pil(paths)
        assert len(frames) == 3
        assert all(isinstance(f, Image.Image) for f in frames)
        assert all(f.mode == "RGB" for f in frames)


class TestExtractAudio:
    @patch("src.utils.video_utils.subprocess.run")
    def test_calls_ffmpeg(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        output = tmp_path / "audio.wav"
        result = extract_audio(Path("/fake/video.mp4"), output)
        assert result == output
        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert "-ar" in args
        assert "16000" in args
