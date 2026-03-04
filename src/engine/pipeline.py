"""전체 분석 파이프라인 오케스트레이션."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from src.engine.segment import Segment, split_video
from src.utils.logger import get_logger
from src.utils.video_utils import extract_audio, extract_frames, load_frames_as_pil

logger = get_logger(__name__)


class Pipeline:
    """AutoClip 전체 처리 파이프라인.

    Step 1: 세그먼트 분할 + 프레임 타임스탬프 계산
    Step 2: 음성 분석 (faster-whisper)
    Step 3: 영상 분석 (Qwen2.5-VL)
    Step 4: 하이라이트 판단 (Qwen2.5 LLM)
    Step 5: 클립 생성 (FFmpeg)
    Step 6: 결과 리포트
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self.analysis_cfg = config.get("analysis", {})
        self.output_cfg = config.get("output", {})
        self.gpu_cfg = config.get("gpu", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, input_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """파이프라인 전체를 실행하고 클립 정보 목록을 반환한다."""
        input_path = Path(input_path)
        output_dir = Path(output_dir)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # Step 1: 세그먼트 분할
            logger.info("Step 1: 영상 세그먼트 분할 중…")
            segments = self._step1_split(input_path)
            logger.info("총 %d개 세그먼트 생성", len(segments))

            # Step 2: 음성 분석
            logger.info("Step 2: 음성 분석 중… (faster-whisper)")
            self._step2_audio(input_path, segments, tmp_dir)

            # Step 3: 영상 분석
            logger.info("Step 3: 영상 분석 중… (Qwen2.5-VL)")
            self._step3_video(input_path, segments, tmp_dir)

            # Step 4: 하이라이트 판단
            logger.info("Step 4: 하이라이트 판단 중… (Qwen2.5 LLM)")
            self._step4_judge(segments)

            # Step 5: 클립 생성
            logger.info("Step 5: 클립 생성 중… (FFmpeg)")
            clips = self._step5_clip(input_path, segments, output_dir)

            # Step 6: 결과 리포트
            logger.info("Step 6: 결과 리포트 생성")
            self._step6_report(clips, output_dir)

        return clips

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _step1_split(self, input_path: Path) -> List[Segment]:
        segment_length = self.analysis_cfg.get("segment_length", 30)
        frames_per_segment = self.analysis_cfg.get("frames_per_segment", 6)
        return split_video(input_path, segment_length, frames_per_segment)

    def _step2_audio(
        self, input_path: Path, segments: List[Segment], tmp_dir: Path
    ) -> None:
        from src.analyzer.audio_analyzer import AudioAnalyzer
        from src.engine.gpu_manager import log_vram, release_model

        audio_path = tmp_dir / "audio.wav"
        extract_audio(input_path, audio_path)

        analyzer = AudioAnalyzer(
            model_size=self.cfg.get("models", {}).get("audio_model", "large-v3"),
            device=self.gpu_cfg.get("device", "cuda"),
            compute_type=self.cfg.get("models", {}).get("compute_type", "float16"),
            language=self.analysis_cfg.get("language", "ko"),
        )
        log_vram("audio 모델 로드 전")
        transcript_segments = analyzer.transcribe(audio_path)
        log_vram("audio 분석 완료")

        # 각 세그먼트에 해당 시간대 텍스트 매핑
        for seg in segments:
            texts = [
                t["text"]
                for t in transcript_segments
                if t["start"] < seg.end and t["end"] > seg.start
            ]
            seg.transcript = " ".join(texts).strip()

        release_model(analyzer.model)
        analyzer.model = None

    def _step3_video(
        self, input_path: Path, segments: List[Segment], tmp_dir: Path
    ) -> None:
        from src.analyzer.video_analyzer import VideoAnalyzer
        from src.engine.gpu_manager import log_vram, release_model

        analyzer = VideoAnalyzer(
            model_name=self.cfg.get("models", {}).get(
                "video_model", "Qwen/Qwen2.5-VL-7B-Instruct"
            ),
            device=self.gpu_cfg.get("device", "cuda"),
        )
        log_vram("video 모델 로드 전")
        analyzer.load()
        log_vram("video 모델 로드 완료")

        frames_dir = tmp_dir / "frames"
        for seg in tqdm(segments, desc="영상 분석"):
            frame_paths = extract_frames(
                input_path, seg.frame_timestamps, frames_dir / f"seg_{seg.index:04d}"
            )
            frames = load_frames_as_pil(frame_paths)
            seg.video_description = analyzer.analyze(frames)

        log_vram("video 분석 완료")
        release_model(analyzer.model)
        analyzer.model = None

    def _step4_judge(self, segments: List[Segment]) -> None:
        from src.analyzer.highlight_judge import HighlightJudge
        from src.engine.gpu_manager import log_vram, release_model

        judge = HighlightJudge(
            model_name=self.cfg.get("models", {}).get(
                "judge_model", "Qwen/Qwen2.5-7B-Instruct"
            ),
            device=self.gpu_cfg.get("device", "cuda"),
        )
        log_vram("judge 모델 로드 전")
        judge.load()
        log_vram("judge 모델 로드 완료")

        threshold = self.analysis_cfg.get("highlight_threshold", 60)
        for seg in tqdm(segments, desc="하이라이트 판단"):
            result = judge.judge(
                seg.video_description,
                seg.transcript,
                seg.start,
                seg.end,
            )
            seg.score = result.get("score", 0)
            seg.is_highlight = result.get("is_highlight", False) or seg.score >= threshold
            seg.reason = result.get("reason", "")
            seg.tags = result.get("tags", [])
            seg.suggested_title = result.get("suggested_title", "")

        log_vram("judge 완료")
        release_model(judge.model)
        judge.model = None

    def _step5_clip(
        self,
        input_path: Path,
        segments: List[Segment],
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        from src.clipper.clip_generator import ClipGenerator

        max_clips = self.analysis_cfg.get("max_clips", 10)
        merge_gap = self.analysis_cfg.get("merge_gap", 8)
        pre_buffer = self.analysis_cfg.get("pre_buffer", 5)
        post_buffer = self.analysis_cfg.get("post_buffer", 3)
        fmt = self.output_cfg.get("format", "shorts")
        include_subtitles = self.output_cfg.get("include_subtitles", True)
        generate_thumbnail = self.output_cfg.get("generate_thumbnail", True)
        generate_compilation = self.output_cfg.get("generate_compilation", True)

        highlights = sorted(
            [s for s in segments if s.is_highlight],
            key=lambda s: s.score,
            reverse=True,
        )[:max_clips]

        if not highlights:
            logger.warning("하이라이트 구간이 감지되지 않았습니다.")
            return []

        # 병합: merge_gap 이내 인접 구간 합치기
        merged = _merge_segments(highlights, merge_gap)

        # 영상 총 길이 (세그먼트 마지막 end가 곧 영상 길이)
        video_duration = segments[-1].end if segments else float("inf")

        generator = ClipGenerator(
            config=self.cfg,
            output_dir=output_dir,
            fmt=fmt,
        )

        clips: List[Dict[str, Any]] = []
        for idx, group in enumerate(merged):
            start = max(0.0, group[0].start - pre_buffer)
            end = min(group[-1].end + post_buffer, video_duration)
            title = group[0].suggested_title or f"highlight_{idx + 1:03d}"
            tags = list({tag for seg in group for tag in seg.tags})
            score = max(seg.score for seg in group)

            clip_path = generator.generate(
                input_path=input_path,
                start=start,
                end=end,
                title=title,
                index=idx,
                include_subtitles=include_subtitles,
                transcript=" ".join(seg.transcript for seg in group),
            )

            if generate_thumbnail:
                from src.clipper.thumbnail import generate_thumbnail as gen_thumb

                gen_thumb(
                    video_path=clip_path,
                    output_path=clip_path.with_suffix(".jpg"),
                    title=title,
                )

            clips.append(
                {
                    "index": idx,
                    "path": str(clip_path),
                    "start": start,
                    "end": end,
                    "score": score,
                    "tags": tags,
                    "title": title,
                }
            )

        if generate_compilation and clips:
            generator.generate_compilation(
                [Path(c["path"]) for c in clips],
                output_dir / "compilation.mp4",
            )

        return clips

    def _step6_report(
        self, clips: List[Dict[str, Any]], output_dir: Path
    ) -> None:
        report_path = output_dir / "report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(clips, f, ensure_ascii=False, indent=2)
        logger.info("결과 리포트 저장: %s", report_path)
        logger.info("생성된 클립 수: %d", len(clips))
        for c in clips:
            logger.info(
                "  [%d] %s (score=%.1f, tags=%s)",
                c["index"],
                c["title"],
                c["score"],
                c["tags"],
            )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _merge_segments(
    highlights: List[Segment], merge_gap: float
) -> List[List[Segment]]:
    """점수 기준 정렬된 하이라이트를 시간 순서로 재정렬한 뒤 인접 구간을 병합한다."""
    sorted_by_time = sorted(highlights, key=lambda s: s.start)
    groups: List[List[Segment]] = []
    for seg in sorted_by_time:
        if groups and seg.start - groups[-1][-1].end <= merge_gap:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    return groups
