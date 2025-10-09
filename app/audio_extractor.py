from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


class AudioExtractor:
    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        self.ffmpeg_path = ffmpeg_path

    def extract(self, video_path: Path, output_dir: Path, sample_rate: int = 16000) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / f"{video_path.stem}.wav"
        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(audio_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_path


__all__ = ["AudioExtractor"]
