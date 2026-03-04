"""HighlightJudge 유닛 테스트 – 실제 모델 없이 mock 사용."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.analyzer.highlight_judge import HighlightJudge, _FALLBACK_RESULT


def _make_torch_mock() -> MagicMock:
    mock_torch = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_torch.no_grad.return_value = ctx
    return mock_torch


class TestHighlightJudgeParseJson:
    def test_valid_json(self):
        raw = json.dumps({
            "score": 75,
            "is_highlight": True,
            "reason": "재미있는 장면",
            "tags": ["funny"],
            "suggested_title": "웃긴 순간",
        })
        result = HighlightJudge._parse_json(raw)
        assert result is not None
        assert result["score"] == 75
        assert result["is_highlight"] is True

    def test_json_in_markdown_block(self):
        raw = '```json\n{"score": 50, "is_highlight": false}\n```'
        result = HighlightJudge._parse_json(raw)
        assert result is not None
        assert result["score"] == 50

    def test_missing_required_keys_returns_none(self):
        raw = '{"reason": "ok"}'
        assert HighlightJudge._parse_json(raw) is None

    def test_invalid_json_returns_none(self):
        assert HighlightJudge._parse_json("이것은 JSON이 아닙니다") is None

    def test_defaults_filled(self):
        raw = '{"score": 80, "is_highlight": true}'
        result = HighlightJudge._parse_json(raw)
        assert result["reason"] == ""
        assert result["tags"] == []
        assert result["suggested_title"] == ""


class TestHighlightJudgeJudge:
    def test_judge_raises_if_not_loaded(self):
        hj = HighlightJudge()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            hj.judge("desc", "transcript", 0.0, 30.0)

    def test_judge_returns_parsed_result(self):
        mock_torch = _make_torch_mock()

        hj = HighlightJudge(max_retries=0)
        hj.model = MagicMock()
        hj.tokenizer = MagicMock()

        valid_response = json.dumps({
            "score": 85,
            "is_highlight": True,
            "reason": "극적인 장면",
            "tags": ["clutch"],
            "suggested_title": "역전 순간",
        })

        hj.tokenizer.apply_chat_template.return_value = "prompt"
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=[1, 5]))
        hj.tokenizer.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        hj.model.generate.return_value = MagicMock()
        hj.tokenizer.batch_decode.return_value = [valid_response]

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = hj.judge("비디오 설명", "음성 텍스트", 0.0, 30.0)
        assert result["score"] == 85
        assert result["is_highlight"] is True
        assert result["tags"] == ["clutch"]

    def test_judge_falls_back_on_parse_failure(self):
        mock_torch = _make_torch_mock()

        hj = HighlightJudge(max_retries=1)
        hj.model = MagicMock()
        hj.tokenizer = MagicMock()

        hj.tokenizer.apply_chat_template.return_value = "prompt"
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=[1, 5]))
        hj.tokenizer.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs
        hj.model.generate.return_value = MagicMock()
        hj.tokenizer.batch_decode.return_value = ["완전히 잘못된 응답"]

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = hj.judge("desc", "transcript", 0.0, 30.0)
        assert result["score"] == _FALLBACK_RESULT["score"]
        assert result["is_highlight"] is False
