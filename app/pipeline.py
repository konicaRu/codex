from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import yaml

from .audio_extractor import AudioExtractor
from .asr_engine import ASREngine
from .llm_client import LLMClient, build_client
from .models import ActionsResult, MinutesResult, PipelineResult, TopicsResult, VideoSegment
from .nlp_merge import merge_ocr_asr
from .ocr_engine import OCREngine
from .render_pdf import PDFRenderer
from .render_pptx import PPTXRenderer
from .video_reader import SceneSegment, VideoSceneSplitter


class Pipeline:
    def __init__(self, config_path: Path) -> None:
        with config_path.open("r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)

        paths = self.config.get("paths", {})
        video_cfg = self.config.get("video", {})
        ocr_cfg = self.config.get("ocr", {})
        asr_cfg = self.config.get("asr", {})
        llm_cfg = self.config.get("llm", {})

        temp_dir = Path(paths.get("temp_dir", "temp"))
        temp_dir.mkdir(parents=True, exist_ok=True)

        self.scene_splitter = VideoSceneSplitter(
            output_dir=temp_dir / "frames",
            histogram_threshold=video_cfg.get("histogram_threshold", 0.35),
            ssim_threshold=video_cfg.get("ssim_threshold", 0.65),
            min_segment_duration=video_cfg.get("min_segment_duration", 4.0),
            max_segment_duration=video_cfg.get("max_segment_duration", 180.0),
            frame_sample_rate=video_cfg.get("frame_sample_rate", 1.0),
            max_slides=video_cfg.get("max_slides"),
        )
        self.ocr_engine = OCREngine(
            lang=ocr_cfg.get("lang", "rus+eng"),
            tesseract_cmd=paths.get("tesseract"),
            psm=ocr_cfg.get("psm", 6),
            oem=ocr_cfg.get("oem", 3),
        )
        self.audio_extractor = AudioExtractor(ffmpeg_path=paths.get("ffmpeg", "ffmpeg"))
        self.asr_engine = ASREngine(
            model_size=asr_cfg.get("model_size", "base"),
            device=asr_cfg.get("device", "auto"),
            compute_type=asr_cfg.get("compute_type", "int8_float16"),
        )
        self.llm_client: LLMClient = build_client(llm_cfg)
        self.temp_dir = temp_dir

    def run(
        self,
        video_path: Path,
        output_format: str = "pdf",
        max_slides: Optional[int] = None,
        asr_language: Optional[str] = None,
    ) -> PipelineResult:
        video_name = video_path.stem
        frames_dir = self.temp_dir / video_name / "frames"
        audio_dir = self.temp_dir / video_name / "audio"
        frames_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        self.scene_splitter.output_dir = frames_dir
        if max_slides is not None:
            self.scene_splitter.max_slides = max_slides

        scene_segments = self.scene_splitter.process(video_path)
        video_segments = self._create_video_segments(scene_segments)

        for segment in video_segments:
            try:
                segment.ocr_text = self.ocr_engine.extract_text(segment.best_frame_path)
            except Exception as exc:  # noqa: BLE001
                segment.ocr_text = f"[OCR error: {exc}]"

        audio_path = self.audio_extractor.extract(video_path, audio_dir)
        asr_segments = self.asr_engine.transcribe(audio_path, language=asr_language)
        self._attach_asr_to_segments(video_segments, asr_segments)

        for segment in video_segments:
            title, merged = merge_ocr_asr(segment.ocr_text, segment.asr_text)
            segment.summary_title = title
            segment.asr_text = merged

        context = {"video": video_name}
        for segment in video_segments:
            result = self.llm_client.summarize_segment(segment, context)
            segment.summary_title = result.title or segment.summary_title
            segment.summary_bullets = result.bullets or [segment.asr_text[:120]]

        topics = self.llm_client.cluster_topics(video_segments)
        actions = self.llm_client.make_action_items(video_segments)
        minutes = self.llm_client.make_global_minutes(video_segments)

        renderer = self._select_renderer(output_format)
        output_path = renderer.render(video_name, video_segments, topics, actions, minutes)

        log_path = self._write_log(video_path, video_segments, topics, actions, minutes, output_path)

        return PipelineResult(
            video_path=video_path,
            segments=video_segments,
            topics=topics,
            actions=actions,
            minutes=minutes,
            output_path=output_path,
            log_path=log_path,
        )

    def _select_renderer(self, output_format: str):
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        if output_format.lower() == "pptx":
            return PPTXRenderer(reports_dir)
        return PDFRenderer(reports_dir)

    def _create_video_segments(self, scene_segments: Iterable[SceneSegment]) -> List[VideoSegment]:
        video_segments: List[VideoSegment] = []
        for idx, scene in enumerate(scene_segments):
            video_segments.append(
                VideoSegment(
                    index=idx,
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    best_frame_path=scene.best_frame_path,
                )
            )
        return video_segments

    def _attach_asr_to_segments(self, segments: List[VideoSegment], asr_segments) -> None:
        for segment in segments:
            texts: List[str] = []
            for asr_segment in asr_segments:
                if asr_segment.end <= segment.start_time:
                    continue
                if asr_segment.start >= segment.end_time:
                    continue
                texts.append(asr_segment.text)
            segment.asr_text = " ".join(texts)

    def _write_log(
        self,
        video_path: Path,
        segments: Iterable[VideoSegment],
        topics: TopicsResult,
        actions: ActionsResult,
        minutes: MinutesResult,
        output_path: Path,
    ) -> Path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{video_path.stem}.json"
        data = {
            "video": str(video_path),
            "output": str(output_path),
            "segments": [
                {
                    "index": seg.index,
                    "start": seg.start_time,
                    "end": seg.end_time,
                    "frame": str(seg.best_frame_path),
                    "ocr": seg.ocr_text,
                    "asr": seg.asr_text,
                    "title": seg.summary_title,
                    "bullets": seg.summary_bullets,
                }
                for seg in segments
            ],
            "topics": topics.topics,
            "topics_summary": topics.summary,
            "actions": actions.actions,
            "minutes": minutes.bullets,
        }
        log_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return log_path


__all__ = ["Pipeline"]
