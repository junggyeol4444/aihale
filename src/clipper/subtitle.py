"""SRT 자막 파일 생성."""

from __future__ import annotations

from pathlib import Path


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
) -> Path:
    """간단한 SRT 자막 파일을 생성한다.

    Whisper 세그먼트 정보가 없는 경우 transcript를 균일하게 나눠서 자막 블록을 만든다.

    Args:
        transcript: 자막으로 사용할 텍스트.
        clip_start: 클립 시작 시간 (오프셋 계산용, 초).
        clip_duration: 클립 길이 (초).
        output_path: 저장할 SRT 파일 경로.
        chars_per_line: 한 블록당 최대 글자 수.
        seconds_per_block: 각 자막 블록의 표시 시간 (초).

    Returns:
        저장된 SRT 파일 경로.
    """
    output_path = Path(output_path)
    words = transcript.split()

    blocks: list[str] = []
    current: list[str] = []
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

    lines: list[str] = []
    for i, block in enumerate(blocks):
        t_start = i * seconds_per_block
        t_end = min((i + 1) * seconds_per_block, clip_duration)
        if t_start >= clip_duration:
            break
        lines.append(str(i + 1))
        lines.append(f"{_fmt_srt_time(t_start)} --> {_fmt_srt_time(t_end)}")
        lines.append(block)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
