"""AudioAnalyzer 유닛 테스트 – 실제 모델 없이 mock 사용."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.analyzer.audio_analyzer import AudioAnalyzer


class TestAudioAnalyzerInit:
    def test_default_values(self):
        aa = AudioAnalyzer()
        assert aa.model_size == "large-v3"
        assert aa.device == "cuda"
        assert aa.compute_type == "float16"
        assert aa.language == "ko"
        assert aa.model is None

    def test_custom_values(self):
        aa = AudioAnalyzer(model_size="base", device="cpu", compute_type="int8", language="en")
        assert aa.model_size == "base"
        assert aa.device == "cpu"
        assert aa.language == "en"


class TestAudioAnalyzerTranscribe:
    def test_transcribe_returns_segments(self, tmp_path):
        mock_model = MagicMock()
        mock_faster_whisper = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model

        seg1 = MagicMock(start=0.0, end=5.0, text="안녕하세요")
        seg2 = MagicMock(start=5.0, end=10.0, text="반갑습니다")
        mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.dict(sys.modules, {"faster_whisper": mock_faster_whisper}):
            aa = AudioAnalyzer(device="cpu")
            result = aa.transcribe(audio_file)

        assert len(result) == 2
        assert result[0] == {"start": 0.0, "end": 5.0, "text": "안녕하세요"}
        assert result[1] == {"start": 5.0, "end": 10.0, "text": "반갑습니다"}

    def test_transcribe_loads_model_once(self, tmp_path):
        mock_model = MagicMock()
        mock_faster_whisper = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model
        mock_model.transcribe.return_value = ([], MagicMock())

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch.dict(sys.modules, {"faster_whisper": mock_faster_whisper}):
            aa = AudioAnalyzer(device="cpu")
            aa.transcribe(audio_file)
            aa.transcribe(audio_file)

        # 두 번 호출해도 WhisperModel 생성자는 한 번만 호출돼야 함
        assert mock_faster_whisper.WhisperModel.call_count == 1
