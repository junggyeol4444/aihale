"""config 유닛 테스트 – YAML 설정 로더."""

from __future__ import annotations

import pytest

from src.utils.config import load_config


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("analysis:\n  segment_length: 30\n", encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg["analysis"]["segment_length"] == 30

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_raises_on_non_dict(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="딕셔너리"):
            load_config(cfg_file)

    def test_loads_default_config(self):
        """실제 config/default.yaml 파일 로드 테스트."""
        from pathlib import Path
        default_path = Path(__file__).parent.parent / "config" / "default.yaml"
        cfg = load_config(default_path)
        assert "models" in cfg
        assert "analysis" in cfg
        assert "output" in cfg
