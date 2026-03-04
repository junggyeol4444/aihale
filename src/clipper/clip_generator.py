"""FFmpeg 기반 클립 생성."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.clipper.subtitle import build_srt

logger = logging.getLogger(__name__)

# 출력 포맷별 기본 해상도 / 최대 길이
_FORMAT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "shorts": {"width": 1080, "height": 1920, "max_duration": 60},
    "tiktok": {"width": 1080, "height": 1920, "max_duration": 60},
    "reel": {"width": 1080, "height": 1920, "max_duration": 90},
    "landscape": {"width": 1920, "height": 1080, "max_duration": 300},
}


def _run_ffmpeg(args: List[str]) -> None:
    """FFmpeg 명령을 실행하고 실패 시 예외를 발생시킨다."""
    cmd = ["ffmpeg", "-y"] + args
    logger.debug("FFmpeg 실행: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg 실행 실패 (code={result.returncode}):\n{result.stderr[-2000:]}"
        )


class ClipGenerator:
    """하이라이트 구간을 FFmpeg로 클립 영상으로 변환한다."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        fmt: str = "shorts",
    ) -> None:
        self.cfg = config
        self.output_dir = Path(output_dir)
        self.fmt = fmt

        # 포맷 설정 결합 (config > 기본값)
        fmt_cfg = _FORMAT_DEFAULTS.get(fmt, _FORMAT_DEFAULTS["shorts"]).copy()
        fmt_cfg.update(config.get("formats", {}).get(fmt, {}))
        self.width: int = int(fmt_cfg["width"])
        self.height: int = int(fmt_cfg["height"])
        self.max_duration: int = int(fmt_cfg["max_duration"])

        out_cfg = config.get("output", {})
        self.video_codec: str = out_cfg.get("video_codec", "libx264")
        self.audio_codec: str = out_cfg.get("audio_codec", "aac")
        self.crf: int = int(out_cfg.get("crf", 23))

    def generate(
        self,
        input_path: Path,
        start: float,
        end: float,
        title: str,
        index: int,
        include_subtitles: bool = True,
        transcript: str = "",
        transcript_segments: list[dict] | None = None,
    ) -> Path:
        """단일 클립을 생성하고 출력 파일 경로를 반환한다."""
        duration = min(end - start, self.max_duration)
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
        out_path = self.output_dir / f"clip_{index:03d}_{safe_title}.mp4"

        # 세로 영상 변환 필터 (landscape이면 그대로 유지)
        if self.fmt == "landscape":
            vf = f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2"
        else:
            vf = (
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{self.height}"
            )

        # 자막 파일 생성
        srt_path: Optional[Path] = None
        if include_subtitles and transcript:
            srt_path = out_path.with_suffix(".srt")
            build_srt(
                transcript, start, duration, srt_path,
                transcript_segments=transcript_segments,
            )
            vf += f",subtitles={srt_path}"

        _run_ffmpeg(
            [
                "-ss", str(start),
                "-i", str(input_path),
                "-t", str(duration),
                "-vf", vf,
                "-c:v", self.video_codec,
                "-crf", str(self.crf),
                "-c:a", self.audio_codec,
                "-movflags", "+faststart",
                str(out_path),
            ]
        )
        logger.info("클립 생성 완료: %s", out_path)
        return out_path

    def generate_compilation(
        self, clip_paths: List[Path], output_path: Path
    ) -> Path:
        """클립들을 이어 붙여 하이라이트 모음 영상을 생성한다."""
        if not clip_paths:
            raise ValueError("클립 목록이 비어 있습니다.")

        # FFmpeg concat demuxer 사용
        list_file = output_path.parent / "_concat_list.txt"
        with list_file.open("w", encoding="utf-8") as f:
            for p in clip_paths:
                f.write(f"file '{p.resolve()}'\n")

        _run_ffmpeg(
            [
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                str(output_path),
            ]
        )
        list_file.unlink(missing_ok=True)
        logger.info("하이라이트 모음 생성 완료: %s", output_path)
        return output_path
