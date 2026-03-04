"""SRT 자막 파일 생성."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt_srt_time(seconds: float) -> str:
    """초를 SRT 타임코드 형식(HH:MM:SS,mmm)으로 변환한다."""
    ms = int((seconds % 1) * 1000)
    total_s = int(seconds)
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(
    transcript: str,
    clip_start: float,
    clip_duration: float,
    output_path: str | Path,
    chars_per_line: int = 20,
    seconds_per_block: float = 3.0,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """간단한 SRT 자막 파일을 생성한다.

    Whisper 세그먼트 정보가 있으면 실제 타임스탬프를 사용하고,
    없으면 transcript를 균일하게 나눠서 자막 블록을 만든다.

    Args:
        transcript: 자막으로 사용할 텍스트.
        clip_start: 클립 시작 시간 (오프셋 계산용, 초).
        clip_duration: 클립 길이 (초).
        output_path: 저장할 SRT 파일 경로.
        chars_per_line: 한 블록당 최대 글자 수 (타임스탬프 없을 때 사용).
        seconds_per_block: 각 자막 블록의 표시 시간 (타임스탬프 없을 때, 초).
        transcript_segments: Whisper 세그먼트 목록
            (각 원소: {"start": float, "end": float, "text": str}).

    Returns:
        저장된 SRT 파일 경로.
    """
    output_path = Path(output_path)

    if transcript_segments:
        lines = _build_srt_from_segments(
            transcript_segments, clip_start, clip_duration
        )
    else:
        lines = _build_srt_uniform(
            transcript, clip_duration, chars_per_line, seconds_per_block
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _build_srt_from_segments(
    segments: List[Dict[str, Any]],
    clip_start: float,
    clip_duration: float,
) -> List[str]:
    """Whisper 세그먼트의 실제 타임스탬프로 SRT 블록을 생성한다."""
    lines: List[str] = []
    clip_end = clip_start + clip_duration
    idx = 1
    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        if not text:
            continue
        # 클립 범위를 벗어나면 건너뜀
        if seg_end <= clip_start or seg_start >= clip_end:
            continue
        # 클립 기준 상대 시간으로 변환
        t_start = max(seg_start - clip_start, 0.0)
        t_end = min(seg_end - clip_start, clip_duration)
        lines.append(str(idx))
        lines.append(f"{_fmt_srt_time(t_start)} --> {_fmt_srt_time(t_end)}")
        lines.append(text)
        lines.append("")
        idx += 1
    return lines


def _build_srt_uniform(
    transcript: str,
    clip_duration: float,
    chars_per_line: int,
    seconds_per_block: float,
) -> List[str]:
    """타임스탬프 없이 균일 블록 기반으로 SRT를 생성한다."""
    words = transcript.split()

    blocks: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        if current_len + len(word) > chars_per_line and current:
            blocks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1
    if current:
        blocks.append(" ".join(current))

    lines: List[str] = []
    for i, block in enumerate(blocks):
        t_start = i * seconds_per_block
        t_end = min((i + 1) * seconds_per_block, clip_duration)
        if t_start >= clip_duration:
            break
        lines.append(str(i + 1))
        lines.append(f"{_fmt_srt_time(t_start)} --> {_fmt_srt_time(t_end)}")
        lines.append(block)
        lines.append("")
    return lines
