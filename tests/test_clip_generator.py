"""ClipGenerator 유닛 테스트 – FFmpeg 호출을 mock 처리."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.clipper.clip_generator import ClipGenerator, _run_ffmpeg
from src.clipper.subtitle import build_srt, _fmt_srt_time


# ---------------------------------------------------------------------------
# subtitle.py 테스트
# ---------------------------------------------------------------------------

class TestFmtSrtTime:
    def test_zero(self):
        assert _fmt_srt_time(0.0) == "00:00:00,000"

    def test_one_hour(self):
        assert _fmt_srt_time(3600.0) == "01:00:00,000"

    def test_with_ms(self):
        assert _fmt_srt_time(1.5) == "00:00:01,500"


class TestBuildSrt:
    def test_creates_file(self, tmp_path):
        srt = tmp_path / "sub.srt"
        build_srt("안녕 하세요 반갑 습니다", clip_start=0.0, clip_duration=10.0, output_path=srt)
        assert srt.exists()
        content = srt.read_text(encoding="utf-8")
        assert "1\n" in content
        assert "-->" in content

    def test_empty_transcript(self, tmp_path):
        srt = tmp_path / "sub.srt"
        build_srt("", clip_start=0.0, clip_duration=10.0, output_path=srt)
        assert srt.exists()
        # 내용이 비어 있어도 파일은 생성됨
        content = srt.read_text(encoding="utf-8")
        assert content == ""


# ---------------------------------------------------------------------------
# clip_generator.py 테스트
# ---------------------------------------------------------------------------

class TestRunFfmpeg:
    @patch("src.clipper.clip_generator.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        _run_ffmpeg(["-i", "input.mp4", "output.mp4"])
        mock_run.assert_called_once()

    @patch("src.clipper.clip_generator.subprocess.run")
    def test_failure_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="error message")
        with pytest.raises(RuntimeError, match="FFmpeg 실행 실패"):
            _run_ffmpeg(["-i", "bad.mp4"])


class TestClipGeneratorInit:
    def test_defaults(self, tmp_path):
        cfg = {"output": {}, "formats": {}, "gpu": {}}
        gen = ClipGenerator(cfg, tmp_path, fmt="shorts")
        assert gen.width == 1080
        assert gen.height == 1920
        assert gen.max_duration == 60

    def test_landscape(self, tmp_path):
        cfg = {"output": {}, "formats": {}, "gpu": {}}
        gen = ClipGenerator(cfg, tmp_path, fmt="landscape")
        assert gen.width == 1920
        assert gen.height == 1080


class TestClipGeneratorGenerate:
    @patch("src.clipper.clip_generator._run_ffmpeg")
    def test_generate_calls_ffmpeg(self, mock_ffmpeg, tmp_path):
        cfg = {"output": {"video_codec": "libx264", "audio_codec": "aac", "crf": 23}, "formats": {}}
        gen = ClipGenerator(cfg, tmp_path, fmt="landscape")
        gen.generate(
            input_path=Path("/fake/video.mp4"),
            start=10.0,
            end=40.0,
            title="test clip",
            index=0,
            include_subtitles=False,
        )
        mock_ffmpeg.assert_called_once()
        args = mock_ffmpeg.call_args[0][0]
        assert str(tmp_path / "clip_000_test clip.mp4") in args

    @patch("src.clipper.clip_generator._run_ffmpeg")
    def test_generate_compilation_calls_ffmpeg(self, mock_ffmpeg, tmp_path):
        cfg = {"output": {}, "formats": {}}
        gen = ClipGenerator(cfg, tmp_path)
        clips = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        gen.generate_compilation(clips, tmp_path / "compilation.mp4")
        mock_ffmpeg.assert_called_once()

    @patch("src.clipper.clip_generator._run_ffmpeg")
    def test_generate_compilation_empty_raises(self, mock_ffmpeg, tmp_path):
        cfg = {"output": {}, "formats": {}}
        gen = ClipGenerator(cfg, tmp_path)
        with pytest.raises(ValueError):
            gen.generate_compilation([], tmp_path / "compilation.mp4")
