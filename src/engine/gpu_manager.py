"""GPU VRAM 관리 – 모델 로드/언로드 헬퍼."""

from __future__ import annotations

import gc
import logging
from typing import Any

logger = logging.getLogger(__name__)


def vram_usage_gb() -> float:
    """현재 GPU VRAM 사용량을 GB 단위로 반환한다. CUDA가 없으면 0을 반환한다."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)
    except ImportError:
        return 0.0


def log_vram(tag: str = "") -> None:
    """현재 VRAM 사용량을 로그로 출력한다."""
    used = vram_usage_gb()
    label = f"[{tag}] " if tag else ""
    logger.info("%sVRAM 사용량: %.2f GB", label, used)


def release_model(model: Any) -> None:
    """모델 객체를 메모리에서 해제하고 VRAM 캐시를 정리한다.

    Args:
        model: 해제할 모델 객체 (None이어도 안전하게 처리).
    """
    if model is None:
        return
    try:
        import torch

        del model
        torch.cuda.empty_cache()
    except ImportError:
        del model
    finally:
        gc.collect()
    log_vram("release_model 완료")


def check_vram_available(required_gb: float, max_vram_gb: float = 9.5) -> bool:
    """필요한 VRAM이 확보 가능한지 확인한다.

    Args:
        required_gb: 필요한 VRAM (GB).
        max_vram_gb: 허용 최대 VRAM (GB, 안전 마진 포함).

    Returns:
        True이면 로드 가능, False이면 불가.
    """
    used = vram_usage_gb()
    available = max_vram_gb - used
    if available < required_gb:
        logger.warning(
            "VRAM 부족: 필요 %.1f GB, 가용 %.1f GB (사용 중 %.1f GB)",
            required_gb,
            available,
            used,
        )
        return False
    return True
