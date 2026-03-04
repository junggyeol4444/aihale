"""썸네일 자동 생성 – FFmpeg + Pillow."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

logger = logging.getLogger(__name__)

_THUMBNAIL_SIZE: Tuple[int, int] = (1280, 720)
_FONT_SIZE = 60
_TEXT_COLOR = (255, 255, 255)
_SHADOW_COLOR = (0, 0, 0)
_SHADOW_OFFSET = (3, 3)


def _capture_frame(video_path: Path, output_path: Path, time_offset: float = 1.0) -> Path:
    """FFmpeg로 영상에서 단일 프레임을 캡처한다."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(time_offset),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )
    return output_path


def _draw_title(img: Image.Image, title: str) -> Image.Image:
    """이미지 하단에 제목 텍스트를 오버레이한다."""
    draw = ImageDraw.Draw(img)
    font: Optional[ImageFont.FreeTypeFont] = None
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", _FONT_SIZE)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # 텍스트 위치: 하단 중앙
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (img.width - text_w) // 2
    y = img.height - text_h - 40

    # 그림자 효과
    draw.text((x + _SHADOW_OFFSET[0], y + _SHADOW_OFFSET[1]), title, font=font, fill=_SHADOW_COLOR)
    draw.text((x, y), title, font=font, fill=_TEXT_COLOR)
    return img


def generate_thumbnail(
    video_path: str | Path,
    output_path: str | Path,
    title: str = "",
    time_offset: float = 1.0,
) -> Path:
    """클립 영상에서 썸네일 이미지를 생성한다.

    Args:
        video_path: 원본 클립 영상 경로.
        output_path: 썸네일 저장 경로 (.jpg 권장).
        title: 썸네일에 오버레이할 제목 텍스트.
        time_offset: 캡처할 프레임의 시간 오프셋 (초).

    Returns:
        저장된 썸네일 파일 경로.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    tmp_frame = output_path.with_suffix(".tmp.png")

    try:
        _capture_frame(video_path, tmp_frame, time_offset)
        img = Image.open(tmp_frame).convert("RGB")

        # 1280x720으로 리사이즈 + 밝기/대비 자동 조정
        img = img.resize(_THUMBNAIL_SIZE, Image.LANCZOS)
        img = ImageEnhance.Brightness(img).enhance(1.1)
        img = ImageEnhance.Contrast(img).enhance(1.2)

        if title:
            img = _draw_title(img, title)

        img.save(str(output_path), "JPEG", quality=90)
        logger.info("썸네일 생성 완료: %s", output_path)
    finally:
        tmp_frame.unlink(missing_ok=True)

    return output_path
