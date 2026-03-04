"""영상 관련 유틸리티 – 프레임 추출 등."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from PIL import Image


def get_video_duration(video_path: str | Path) -> float:
    """FFprobe를 사용해 영상 길이(초)를 반환한다."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def extract_frames(
    video_path: str | Path,
    timestamps: List[float],
    output_dir: str | Path,
) -> List[Path]:
    """지정한 타임스탬프(초)에서 프레임을 PNG 파일로 추출한다.

    Returns:
        추출된 이미지 파일 경로 목록 (타임스탬프 순서와 동일).
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for i, ts in enumerate(timestamps):
        out_path = output_dir / f"frame_{i:05d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss", str(ts),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(out_path),
            ],
            capture_output=True,
            check=True,
        )
        paths.append(out_path)
    return paths


def load_frames_as_pil(frame_paths: List[Path]) -> List[Image.Image]:
    """이미지 파일 목록을 PIL Image 목록으로 변환한다."""
    return [Image.open(p).convert("RGB") for p in frame_paths]


def sample_timestamps(
    start: float,
    end: float,
    n_frames: int,
) -> List[float]:
    """[start, end] 구간에서 n_frames 개의 균일 타임스탬프를 반환한다."""
    if n_frames <= 0:
        return []
    if n_frames == 1:
        return [(start + end) / 2]
    step = (end - start) / (n_frames - 1)
    return [start + i * step for i in range(n_frames)]


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
) -> Path:
    """영상에서 오디오 트랙을 WAV 파일로 추출한다."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )
    return output_path
