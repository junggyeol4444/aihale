"""영상을 세그먼트 단위로 분할하는 로직."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from src.utils.video_utils import get_video_duration, sample_timestamps


@dataclass
class Segment:
    """단일 영상 세그먼트를 나타낸다."""

    index: int
    start: float          # 시작 시간 (초)
    end: float            # 종료 시간 (초)
    frame_timestamps: List[float] = field(default_factory=list)
    video_description: str = ""
    transcript: str = ""
    transcript_segments: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    is_highlight: bool = False
    reason: str = ""
    tags: List[str] = field(default_factory=list)
    suggested_title: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def split_video(
    video_path: str | Path,
    segment_length: int = 30,
    frames_per_segment: int = 6,
) -> List[Segment]:
    """영상을 *segment_length* 초 단위 세그먼트로 분할한다.

    각 세그먼트마다 균일하게 *frames_per_segment* 개의 프레임 타임스탬프를 계산한다.

    Returns:
        Segment 목록 (시간 순서).
    """
    video_path = Path(video_path)
    total_duration = get_video_duration(video_path)

    segments: List[Segment] = []
    start = 0.0
    idx = 0
    while start < total_duration:
        end = min(start + segment_length, total_duration)
        timestamps = sample_timestamps(start, end, frames_per_segment)
        seg = Segment(
            index=idx,
            start=start,
            end=end,
            frame_timestamps=timestamps,
        )
        segments.append(seg)
        start = end
        idx += 1

    return segments
