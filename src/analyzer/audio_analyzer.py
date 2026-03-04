"""faster-whisper를 사용한 음성→텍스트 변환."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """faster-whisper 모델로 음성 파일을 텍스트로 변환한다."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ko",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        if self.model is None:
            from faster_whisper import WhisperModel

            logger.info("AudioAnalyzer 모델 로드 중: %s", self.model_size)
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("AudioAnalyzer 모델 로드 완료")

    def transcribe(self, audio_path: str | Path) -> List[Dict[str, Any]]:
        """오디오 파일을 텍스트로 변환하고 타임스탬프 포함 세그먼트 목록을 반환한다.

        Returns:
            각 원소: {"start": float, "end": float, "text": str}
        """
        self._ensure_loaded()
        audio_path = Path(audio_path)
        logger.info("음성 변환 중: %s", audio_path)

        segments, _info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=5,
            word_timestamps=True,
        )

        result: List[Dict[str, Any]] = []
        for seg in segments:
            result.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                }
            )

        logger.info("음성 변환 완료. 총 %d개 구간", len(result))
        return result
