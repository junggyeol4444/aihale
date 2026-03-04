"""logger 유닛 테스트."""

from __future__ import annotations

import logging

from src.utils.logger import get_logger


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_sets_level(self):
        logger = get_logger("test_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_has_handler(self):
        logger = get_logger("test_handler")
        assert len(logger.handlers) >= 1

    def test_reuses_existing_handlers(self):
        name = "test_reuse_unique"
        logger1 = get_logger(name)
        handler_count = len(logger1.handlers)
        logger2 = get_logger(name)
        assert logger1 is logger2
        assert len(logger2.handlers) == handler_count
