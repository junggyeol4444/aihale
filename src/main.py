"""CLI 진입점 – argparse 기반 AutoClip 커맨드."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoclip",
        description="로컬 AI 기반 생방송 자동 하이라이트 클리퍼",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="VOD 영상을 분석하여 하이라이트 클립 생성")
    analyze.add_argument(
        "--input", "-i", required=True, type=Path, help="분석할 VOD 영상 경로"
    )
    analyze.add_argument(
        "--output", "-o", required=True, type=Path, help="클립 출력 디렉토리"
    )
    analyze.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "default.yaml",
        help="설정 파일 경로 (기본: config/default.yaml)",
    )
    analyze.add_argument(
        "--segment-length",
        type=int,
        default=None,
        help="세그먼트 길이(초) (기본: config에서 읽음)",
    )
    analyze.add_argument(
        "--top",
        type=int,
        default=None,
        help="생성할 클립 수 (기본: config에서 읽음)",
    )
    analyze.add_argument(
        "--format",
        choices=["shorts", "tiktok", "reel", "landscape"],
        default=None,
        help="출력 포맷 (기본: config에서 읽음)",
    )
    analyze.add_argument(
        "--no-subtitle",
        action="store_true",
        help="자막을 포함하지 않음",
    )
    analyze.add_argument(
        "--no-thumbnail",
        action="store_true",
        help="썸네일을 생성하지 않음",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        from src.utils.config import load_config
        from src.utils.logger import get_logger
        from src.engine.pipeline import Pipeline

        logger = get_logger("autoclip")

        if not args.input.exists():
            logger.error("입력 파일을 찾을 수 없습니다: %s", args.input)
            return 1

        args.output.mkdir(parents=True, exist_ok=True)

        cfg = load_config(args.config)

        # CLI 옵션으로 설정 덮어쓰기
        if args.segment_length is not None:
            cfg["analysis"]["segment_length"] = args.segment_length
        if args.top is not None:
            cfg["analysis"]["max_clips"] = args.top
        if args.format is not None:
            cfg["output"]["format"] = args.format
        if args.no_subtitle:
            cfg["output"]["include_subtitles"] = False
        if args.no_thumbnail:
            cfg["output"]["generate_thumbnail"] = False

        logger.info("분석 시작: %s", args.input)
        pipeline = Pipeline(cfg)
        pipeline.run(input_path=args.input, output_dir=args.output)
        logger.info("완료. 클립 저장 위치: %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
