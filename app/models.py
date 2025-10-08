from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class VideoSegment:
    index: int
    start_time: float
    end_time: float
    best_frame_path: Path
    ocr_text: str = ""
    asr_text: str = ""
    summary_title: str = ""
    summary_bullets: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SegmentLLMResult:
    title: str
    bullets: List[str]


@dataclass
class TopicsResult:
    topics: List[str]
    summary: str


@dataclass
class ActionsResult:
    actions: List[str]


@dataclass
class MinutesResult:
    bullets: List[str]


@dataclass
class PipelineResult:
    video_path: Path
    segments: List[VideoSegment]
    topics: Optional[TopicsResult] = None
    actions: Optional[ActionsResult] = None
    minutes: Optional[MinutesResult] = None
    output_path: Optional[Path] = None
    log_path: Optional[Path] = None

