"""Qwen2.5-7B-Instruct를 사용한 하이라이트 최종 판단."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """다음은 생방송의 한 구간({start_time} ~ {end_time})에 대한 분석 결과입니다.

[영상 분석]: {video_description}
[스트리머 음성]: {transcript}

이 구간이 하이라이트(클립으로 만들 만한 재미있거나 인상적인 순간)인지 판단해주세요.

다음 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{{
  "score": <0~100 정수>,
  "is_highlight": <true 또는 false>,
  "reason": "<판단 이유>",
  "tags": ["funny", "clutch", "fail", "emotional", "skill", "reaction" 중 해당하는 것],
  "suggested_title": "<클립 제목 제안>"
}}"""

_FALLBACK_RESULT: Dict[str, Any] = {
    "score": 0,
    "is_highlight": False,
    "reason": "JSON 파싱 실패",
    "tags": [],
    "suggested_title": "",
}


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class HighlightJudge:
    """Qwen2.5 텍스트 LLM으로 세그먼트의 하이라이트 여부를 판단한다."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_retries: int = 2,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_retries = max_retries
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    def load(self) -> None:
        """모델과 토크나이저를 GPU에 로드한다."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("HighlightJudge 모델 로드 중: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        logger.info("HighlightJudge 모델 로드 완료")

    def judge(
        self,
        video_description: str,
        transcript: str,
        start: float,
        end: float,
    ) -> Dict[str, Any]:
        """세그먼트를 판단하고 결과 딕셔너리를 반환한다.

        JSON 파싱 실패 시 *max_retries* 횟수만큼 재시도한다.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load()를 먼저 호출하세요.")

        prompt = _PROMPT_TEMPLATE.format(
            start_time=_fmt_time(start),
            end_time=_fmt_time(end),
            video_description=video_description or "(설명 없음)",
            transcript=transcript or "(음성 없음)",
        )

        for attempt in range(self.max_retries + 1):
            raw = self._generate(prompt)
            result = self._parse_json(raw)
            if result is not None:
                return result
            logger.warning(
                "JSON 파싱 실패 (시도 %d/%d). 응답: %s",
                attempt + 1,
                self.max_retries + 1,
                raw[:200],
            )

        return dict(_FALLBACK_RESULT)

    def _generate(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """텍스트에서 JSON 객체를 추출한다."""
        # 마크다운 코드 블록 제거
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        # 첫 번째 { ... } 블록 추출
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
            # 필수 키 검증
            if "score" not in data or "is_highlight" not in data:
                return None
            # 타입 정규화
            data["score"] = int(data.get("score", 0))
            data["is_highlight"] = bool(data.get("is_highlight", False))
            data.setdefault("reason", "")
            data.setdefault("tags", [])
            data.setdefault("suggested_title", "")
            return data
        except (json.JSONDecodeError, ValueError):
            return None
