"""Pipeline 유닛 테스트 – 각 스텝을 mock 처리."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.engine.pipeline import Pipeline, _merge_segments
from src.engine.segment import Segment


def _make_segment(index: int, start: float, end: float, score: float = 0.0, is_highlight: bool = False) -> Segment:
    seg = Segment(index=index, start=start, end=end)
    seg.score = score
    seg.is_highlight = is_highlight
    return seg


class TestMergeSegments:
    def test_empty(self):
        assert _merge_segments([], merge_gap=8) == []

    def test_single(self):
        seg = _make_segment(0, 0, 30, is_highlight=True)
        groups = _merge_segments([seg], merge_gap=8)
        assert len(groups) == 1
        assert groups[0] == [seg]

    def test_merge_adjacent(self):
        seg1 = _make_segment(0, 0, 30)
        seg2 = _make_segment(1, 33, 60)   # gap = 3초 < 8초 → 병합
        groups = _merge_segments([seg1, seg2], merge_gap=8)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_no_merge_far_apart(self):
        seg1 = _make_segment(0, 0, 30)
        seg2 = _make_segment(1, 50, 80)   # gap = 20초 > 8초 → 분리
        groups = _merge_segments([seg1, seg2], merge_gap=8)
        assert len(groups) == 2

    def test_time_ordering(self):
        # 점수 기준으로 섞인 입력이 시간 순으로 정렬되는지 확인
        seg1 = _make_segment(0, 60, 90, score=90)
        seg2 = _make_segment(1, 0, 30, score=70)
        groups = _merge_segments([seg1, seg2], merge_gap=8)
        assert groups[0][0].start == 0.0


class TestPipelineInit:
    def test_init_stores_config(self):
        cfg = {"analysis": {"segment_length": 60}, "output": {}, "gpu": {}}
        p = Pipeline(cfg)
        assert p.cfg == cfg
        assert p.analysis_cfg["segment_length"] == 60


class TestPipelineStep6Report:
    def test_report_creates_json(self, tmp_path):
        cfg = {"analysis": {}, "output": {}, "gpu": {}}
        p = Pipeline(cfg)
        clips = [{"index": 0, "title": "test", "score": 80.0, "tags": [], "path": "/tmp/a.mp4", "start": 0.0, "end": 30.0}]
        p._step6_report(clips, tmp_path)
        import json
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert len(report) == 1
        assert report[0]["title"] == "test"
