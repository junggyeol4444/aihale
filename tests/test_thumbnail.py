"""thumbnail 유닛 테스트 – FFmpeg 호출을 mock 처리."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.clipper.thumbnail import _capture_frame, _draw_title, generate_thumbnail


class TestCaptureFrame:
    @patch("src.clipper.thumbnail.subprocess.run")
    def test_calls_ffmpeg(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = _capture_frame(Path("/fake/clip.mp4"), Path("/tmp/frame.png"), 1.0)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert "-ss" in args


class TestDrawTitle:
    def test_returns_image(self):
        img = Image.new("RGB", (1280, 720), color=(128, 128, 128))
        result = _draw_title(img, "테스트 제목")
        assert isinstance(result, Image.Image)
        assert result.size == (1280, 720)

    def test_empty_title(self):
        img = Image.new("RGB", (1280, 720))
        result = _draw_title(img, "")
        assert isinstance(result, Image.Image)


class TestGenerateThumbnail:
    @patch("src.clipper.thumbnail._capture_frame")
    def test_generates_thumbnail(self, mock_capture, tmp_path):
        # _capture_frame이 실제 프레임 파일을 생성하도록 mock
        frame_path = tmp_path / "output.tmp.png"
        Image.new("RGB", (1920, 1080), color=(100, 100, 100)).save(str(frame_path))
        mock_capture.return_value = frame_path

        output_path = tmp_path / "output.jpg"
        result = generate_thumbnail(
            video_path=tmp_path / "clip.mp4",
            output_path=output_path,
            title="테스트 썸네일",
        )
        assert result == output_path
        assert output_path.exists()

        # 결과 이미지가 올바른 크기인지 확인
        img = Image.open(output_path)
        assert img.size == (1280, 720)

    @patch("src.clipper.thumbnail._capture_frame")
    def test_thumbnail_without_title(self, mock_capture, tmp_path):
        frame_path = tmp_path / "output.tmp.png"
        Image.new("RGB", (1920, 1080)).save(str(frame_path))
        mock_capture.return_value = frame_path

        output_path = tmp_path / "output.jpg"
        result = generate_thumbnail(
            video_path=tmp_path / "clip.mp4",
            output_path=output_path,
            title="",
        )
        assert output_path.exists()

    @patch("src.clipper.thumbnail._capture_frame")
    def test_tmp_frame_cleaned_up(self, mock_capture, tmp_path):
        frame_path = tmp_path / "output.tmp.png"
        Image.new("RGB", (640, 480)).save(str(frame_path))
        mock_capture.return_value = frame_path

        output_path = tmp_path / "output.jpg"
        generate_thumbnail(
            video_path=tmp_path / "clip.mp4",
            output_path=output_path,
        )
        # 임시 파일이 정리되었는지 확인
        assert not frame_path.exists()
