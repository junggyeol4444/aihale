"""간편 실행 스크립트 – src.main을 직접 호출."""

from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
