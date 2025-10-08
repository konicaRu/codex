from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from faster_whisper import WhisperModel


class TranscriptionSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text

    def __repr__(self) -> str:
        return f"TranscriptionSegment(start={self.start}, end={self.end}, text={self.text!r})"


class ASREngine:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8_float16",
    ) -> None:
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> List[TranscriptionSegment]:
        segments: List[TranscriptionSegment] = []
        options = {}
        if language:
            options["language"] = language
        transcription, _ = self.model.transcribe(str(audio_path), **options)
        for segment in transcription:
            segments.append(TranscriptionSegment(segment.start, segment.end, segment.text.strip()))
        return segments

    @staticmethod
    def merge_segments(segments: Iterable[TranscriptionSegment]) -> str:
        return " ".join(seg.text for seg in segments if seg.text)


__all__ = ["ASREngine", "TranscriptionSegment"]
