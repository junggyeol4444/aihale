"""gpu_manager 유닛 테스트 – CUDA 없는 환경에서 mock 사용."""

from __future__ import annotations

import gc
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.engine.gpu_manager import check_vram_available, log_vram, release_model, vram_usage_gb


class TestVramUsageGb:
    def test_returns_float_when_cuda_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2 * (1024 ** 3)  # 2 GB

        with patch.dict(sys.modules, {"torch": mock_torch}):
            # reload 없이 함수 내부에서 import 하므로 동작
            result = vram_usage_gb()
        assert result == pytest.approx(2.0)

    def test_returns_zero_when_no_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = vram_usage_gb()
        assert result == 0.0

    def test_returns_zero_when_torch_missing(self):
        with patch.dict(sys.modules, {"torch": None}):
            result = vram_usage_gb()
        assert result == 0.0


class TestLogVram:
    @patch("src.engine.gpu_manager.vram_usage_gb", return_value=3.5)
    def test_log_with_tag(self, mock_usage, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            log_vram("test_tag")
        assert "test_tag" in caplog.text
        assert "3.50" in caplog.text

    @patch("src.engine.gpu_manager.vram_usage_gb", return_value=0.0)
    def test_log_without_tag(self, mock_usage, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            log_vram()
        assert "VRAM" in caplog.text


class TestReleaseModel:
    def test_release_none_is_safe(self):
        # None을 전달해도 에러 없이 처리
        release_model(None)

    @patch("src.engine.gpu_manager.log_vram")
    def test_release_model_clears_cache(self, mock_log):
        mock_torch = MagicMock()
        model = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            release_model(model)

        mock_torch.cuda.empty_cache.assert_called_once()
        mock_log.assert_called_once()


class TestCheckVramAvailable:
    @patch("src.engine.gpu_manager.vram_usage_gb", return_value=2.0)
    def test_enough_vram(self, mock_usage):
        assert check_vram_available(required_gb=5.0, max_vram_gb=9.5) is True

    @patch("src.engine.gpu_manager.vram_usage_gb", return_value=8.0)
    def test_not_enough_vram(self, mock_usage):
        assert check_vram_available(required_gb=5.0, max_vram_gb=9.5) is False

    @patch("src.engine.gpu_manager.vram_usage_gb", return_value=0.0)
    def test_exact_boundary(self, mock_usage):
        assert check_vram_available(required_gb=9.5, max_vram_gb=9.5) is True
