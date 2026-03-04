"""main.py CLI 유닛 테스트."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.main import _build_parser, main


class TestBuildParser:
    def test_analyze_command(self):
        parser = _build_parser()
        args = parser.parse_args(["analyze", "--input", "video.mp4", "--output", "clips/"])
        assert args.command == "analyze"
        assert args.input == Path("video.mp4")
        assert args.output == Path("clips/")

    def test_format_choices(self):
        parser = _build_parser()
        args = parser.parse_args(["analyze", "-i", "v.mp4", "-o", "out/", "--format", "tiktok"])
        assert args.format == "tiktok"

    def test_no_subtitle_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["analyze", "-i", "v.mp4", "-o", "out/", "--no-subtitle"])
        assert args.no_subtitle is True

    def test_no_thumbnail_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["analyze", "-i", "v.mp4", "-o", "out/", "--no-thumbnail"])
        assert args.no_thumbnail is True

    def test_optional_params_default_none(self):
        parser = _build_parser()
        args = parser.parse_args(["analyze", "-i", "v.mp4", "-o", "out/"])
        assert args.segment_length is None
        assert args.top is None
        assert args.format is None


class TestMain:
    def test_missing_input_file(self, tmp_path):
        result = main(["analyze", "-i", str(tmp_path / "nonexistent.mp4"), "-o", str(tmp_path / "out")])
        assert result == 1

    @patch("src.engine.pipeline.Pipeline")
    @patch("src.utils.config.load_config")
    def test_runs_pipeline(self, mock_load_config, mock_pipeline_cls, tmp_path):
        input_file = tmp_path / "video.mp4"
        input_file.write_bytes(b"fake")
        output_dir = tmp_path / "output"

        mock_load_config.return_value = {
            "analysis": {"segment_length": 30, "max_clips": 10},
            "output": {"format": "shorts", "include_subtitles": True, "generate_thumbnail": True},
            "gpu": {},
        }
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        result = main(["analyze", "-i", str(input_file), "-o", str(output_dir)])
        assert result == 0
        mock_pipeline.run.assert_called_once()

    @patch("src.engine.pipeline.Pipeline")
    @patch("src.utils.config.load_config")
    def test_cli_overrides_config(self, mock_load_config, mock_pipeline_cls, tmp_path):
        input_file = tmp_path / "video.mp4"
        input_file.write_bytes(b"fake")

        cfg = {
            "analysis": {"segment_length": 30, "max_clips": 10},
            "output": {"format": "shorts", "include_subtitles": True, "generate_thumbnail": True},
            "gpu": {},
        }
        mock_load_config.return_value = cfg
        mock_pipeline_cls.return_value = MagicMock()

        main([
            "analyze", "-i", str(input_file), "-o", str(tmp_path / "out"),
            "--segment-length", "60", "--top", "5", "--format", "tiktok",
            "--no-subtitle", "--no-thumbnail",
        ])

        assert cfg["analysis"]["segment_length"] == 60
        assert cfg["analysis"]["max_clips"] == 5
        assert cfg["output"]["format"] == "tiktok"
        assert cfg["output"]["include_subtitles"] is False
        assert cfg["output"]["generate_thumbnail"] is False
