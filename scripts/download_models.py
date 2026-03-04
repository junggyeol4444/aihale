"""모델 다운로드 스크립트 – huggingface_hub 사용."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_model(repo_id: str, cache_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"다운로드 중: {repo_id} → {cache_dir}")
    path = snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir))
    print(f"완료: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoClip AI 모델 다운로드")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "autoclip" / "models",
        help="모델 캐시 디렉토리 (기본: ~/.cache/autoclip/models)",
    )
    parser.add_argument(
        "--model",
        choices=["video", "audio", "judge", "all"],
        default="all",
        help="다운로드할 모델 (기본: all)",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "video": "Qwen/Qwen2.5-VL-7B-Instruct",
        "judge": "Qwen/Qwen2.5-7B-Instruct",
    }

    if args.model == "audio":
        print("faster-whisper large-v3 모델은 첫 실행 시 자동으로 다운로드됩니다.")
        print("또는 다음 명령으로 직접 다운로드할 수 있습니다:")
        print("  python -c \"from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu')\"")
        return 0

    to_download = list(models.values()) if args.model == "all" else [models[args.model]]
    for repo_id in to_download:
        try:
            download_model(repo_id, args.cache_dir)
        except Exception as e:  # noqa: BLE001
            print(f"오류: {repo_id} 다운로드 실패: {e}", file=sys.stderr)
            return 1

    print("\n모든 모델 다운로드 완료!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
