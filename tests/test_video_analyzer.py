"""VideoAnalyzer 유닛 테스트 – 실제 모델 없이 mock 사용."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.analyzer.video_analyzer import VideoAnalyzer


def _make_dummy_frames(n: int = 3) -> list[Image.Image]:
    return [Image.new("RGB", (64, 64), color=(i * 80, 0, 0)) for i in range(n)]


def _make_torch_mock() -> MagicMock:
    mock_torch = MagicMock()
    mock_torch.float16 = "float16"
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_torch.no_grad.return_value = ctx
    return mock_torch


def _make_transformers_mock() -> MagicMock:
    mock_transformers = MagicMock()
    return mock_transformers


class TestVideoAnalyzerInit:
    def test_default_values(self):
        va = VideoAnalyzer()
        assert va.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert va.device == "cuda"
        assert va.model is None
        assert va.processor is None

    def test_custom_values(self):
        va = VideoAnalyzer(model_name="my/model", device="cpu")
        assert va.model_name == "my/model"
        assert va.device == "cpu"


class TestVideoAnalyzerLoad:
    def test_load_calls_from_pretrained(self):
        mock_torch = _make_torch_mock()
        mock_transformers = _make_transformers_mock()
        mock_model = MagicMock()
        mock_proc = MagicMock()
        mock_transformers.Qwen2VLForConditionalGeneration.from_pretrained.return_value = mock_model
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_proc

        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
            va = VideoAnalyzer(device="cpu")
            va.load()

        mock_transformers.Qwen2VLForConditionalGeneration.from_pretrained.assert_called_once()
        mock_transformers.AutoProcessor.from_pretrained.assert_called_once()
        assert va.model is mock_model
        assert va.processor is mock_proc


class TestVideoAnalyzerAnalyze:
    def test_analyze_raises_if_not_loaded(self):
        va = VideoAnalyzer()
        frames = _make_dummy_frames()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            va.analyze(frames)

    def test_analyze_returns_string(self):
        mock_torch = _make_torch_mock()

        va = VideoAnalyzer(device="cpu")
        va.model = MagicMock()
        va.processor = MagicMock()

        dummy_text = "chat template result"
        va.processor.apply_chat_template.return_value = dummy_text
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=[1, 10]))
        va.processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        dummy_ids = MagicMock()
        va.model.generate.return_value = dummy_ids
        dummy_ids.__getitem__ = MagicMock(return_value=MagicMock())
        va.processor.batch_decode.return_value = ["  장면 설명 텍스트  "]

        frames = _make_dummy_frames(2)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = va.analyze(frames)
        assert isinstance(result, str)
        assert result == "장면 설명 텍스트"
