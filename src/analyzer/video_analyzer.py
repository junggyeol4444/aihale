"""Qwen2.5-VL-7B-Instruct를 사용한 영상 프레임 분석."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

_PROMPT = (
    "이 영상 프레임들을 분석해줘. "
    "게임 플레이, 스트리머 반응, 특별한 이벤트 등을 설명해줘. "
    "하이라이트가 될 만한 순간이 있다면 구체적으로 설명해줘."
)


class VideoAnalyzer:
    """Qwen2.5-VL 모델로 영상 세그먼트의 프레임들을 분석한다."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None

    def load(self) -> None:
        """모델과 프로세서를 GPU에 로드한다."""
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        logger.info("VideoAnalyzer 모델 로드 중: %s", self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        logger.info("VideoAnalyzer 모델 로드 완료")

    def analyze(self, frames: List[Image.Image], prompt: str = _PROMPT) -> str:
        """프레임 목록을 분석하여 장면 설명 텍스트를 반환한다.

        Args:
            frames: PIL Image 목록 (세그먼트에서 샘플링한 프레임).
            prompt: 모델에 전달할 한국어 프롬프트.

        Returns:
            장면 설명 문자열.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load()를 먼저 호출하세요.")

        import torch

        # Qwen2.5-VL 멀티모달 메시지 구성
        content: List[Any] = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=frames,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        result = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return result.strip()
