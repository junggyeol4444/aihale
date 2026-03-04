# AutoClip – AI 생방송 자동 하이라이트 클리퍼

로컬 AI 모델이 생방송 VOD 영상을 **직접 시청·분석**하여 하이라이트 구간을 자동 감지하고, 클립 영상을 자동 생성하는 프로그램.

## 특징

- **로컬 AI 전용**: 외부 API 비용 0원. 모든 AI 모델을 로컬 GPU에서 실행.
- **콘텐츠 이해 기반**: 채팅 수·소리 크기 같은 단순 신호가 아니라 AI가 영상 내용 자체를 분석.
- **하꼬 스트리머 지원**: 시청자 수와 무관하게 하이라이트 감지.
- **멀티 포맷 출력**: YouTube Shorts / TikTok / Instagram Reels / 가로형 자동 변환.
- **자막 자동 생성**: Whisper 결과 기반 SRT 자막 삽입.
- **썸네일 자동 생성**: 하이라이트 피크 순간 캡처 + 제목 텍스트 오버레이.
- **하이라이트 모음 영상**: 상위 클립들을 이어붙인 컴파일레이션 자동 생성.

## 사용 AI 모델 (전부 로컬, 무료)

| 역할 | 모델 | VRAM |
|------|------|------|
| 영상 분석 | `Qwen/Qwen2.5-VL-7B-Instruct` | ~8 GB |
| 음성→텍스트 | `faster-whisper` (`large-v3`) | ~2–3 GB |
| 하이라이트 판단 | `Qwen/Qwen2.5-7B-Instruct` | ~8 GB |

> **VRAM 관리**: 3개 모델을 순차적으로 로드/언로드하며 RTX 3080 (10 GB) 내에서 동작합니다.

## 요구사항

- Python 3.10+
- NVIDIA GPU (RTX 3080 10 GB 이상 권장)
- CUDA 12.x
- FFmpeg (시스템에 설치 필요)
- 디스크 여유 공간 약 30 GB (모델 다운로드용)

## 설치

```bash
# 1. 저장소 클론
git clone https://github.com/junggyeol4444/aihale.git
cd aihale

# 2. 패키지 설치
pip install -e .
# 또는
pip install -r requirements.txt

# 3. AI 모델 다운로드 (최초 1회)
python scripts/download_models.py
```

## 사용법

### 기본 사용

```bash
python -m src.main analyze --input ./stream.mp4 --output ./clips/
```

### 세부 옵션 지정

```bash
python -m src.main analyze \
  --input ./stream.mp4 \
  --output ./clips/ \
  --format shorts \
  --top 5 \
  --segment-length 30
```

### 자막·썸네일 제외

```bash
python -m src.main analyze \
  --input ./stream.mp4 \
  --output ./clips/ \
  --no-subtitle \
  --no-thumbnail
```

### 모든 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input` | 분석할 VOD 영상 경로 | (필수) |
| `--output` | 클립 출력 디렉토리 | (필수) |
| `--config` | 설정 파일 경로 | `config/default.yaml` |
| `--segment-length` | 세그먼트 길이 (초) | 30 |
| `--top` | 생성할 클립 수 | 10 |
| `--format` | 출력 포맷 (`shorts` / `tiktok` / `reel` / `landscape`) | `shorts` |
| `--no-subtitle` | 자막 미포함 | 자막 포함 |
| `--no-thumbnail` | 썸네일 미생성 | 썸네일 생성 |

## 처리 파이프라인

```
Step 1  영상 세그먼트 분할 (기본 30초 단위)
        └─ 3시간 방송 → 360개 세그먼트, 각 6프레임 샘플링

Step 2  음성 분석 (faster-whisper large-v3)
        └─ 전체 오디오 → 타임스탬프 포함 텍스트 변환

Step 3  영상 분석 (Qwen2.5-VL-7B)
        └─ 각 세그먼트 프레임 → 장면 설명 텍스트 생성

Step 4  하이라이트 판단 (Qwen2.5-7B)
        └─ 영상 설명 + 음성 텍스트 → 점수(0~100) / 태그 / 제목 제안

Step 5  클립 생성 (FFmpeg)
        └─ 상위 N개 구간 추출, 세로 변환, 자막·썸네일 생성

Step 6  결과 리포트 (report.json)
        └─ 각 클립 정보, 점수, 태그 요약
```

## 설정

`config/default.yaml`에서 모든 파라미터를 조정할 수 있습니다.

```yaml
analysis:
  segment_length: 30          # 세그먼트 길이(초)
  frames_per_segment: 6       # 세그먼트당 샘플 프레임 수
  highlight_threshold: 60     # 하이라이트 판정 임계값 (0~100)
  max_clips: 10               # 최대 클립 수
  merge_gap: 8                # 병합 간격(초)
  pre_buffer: 5               # 클립 앞 여유 시간(초)
  post_buffer: 3              # 클립 뒤 여유 시간(초)
  language: "ko"              # 음성 인식 언어

output:
  format: "shorts"            # shorts / tiktok / reel / landscape
  include_subtitles: true
  generate_thumbnail: true
  generate_compilation: true

gpu:
  device: "cuda"
  max_vram_usage: 9.5         # GB 단위 안전 마진
  auto_offload: true
```

## 처리 시간 (참고)

RTX 3080 기준:

| 영상 길이 | 예상 처리 시간 |
|-----------|---------------|
| 1시간 방송 | ~40분–1시간 |
| 3시간 방송 | ~2–3시간 |

## 출력 파일

```
clips/
├── clip_000_역전_순간.mp4      # 클립 영상
├── clip_000_역전_순간.srt      # 자막 파일
├── clip_000_역전_순간.jpg      # 썸네일
├── clip_001_...
├── ...
├── compilation.mp4             # 하이라이트 모음 영상
└── report.json                 # 결과 리포트
```

## 개발 / 테스트

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## 라이선스

MIT License