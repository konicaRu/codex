from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm

from .pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline video meeting minutes generator")
    parser.add_argument("--input", required=True, help="Путь к видеофайлу или папке с видео")
    parser.add_argument("--config", default="config.yaml", help="Путь к config.yaml")
    parser.add_argument("--format", choices=["pdf", "pptx"], default="pdf")
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--lang", default=None, help="Язык для OCR/ASR, например ru+en")
    parser.add_argument(
        "--model",
        default=None,
        help="Имя модели в LM Studio, должно соответствовать config.yaml",
    )
    return parser.parse_args()


def collect_videos(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    videos: List[Path] = []
    for ext in ("*.mp4", "*.mkv", "*.mov"):
        videos.extend(input_path.rglob(ext))
    return videos


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pipeline = Pipeline(Path(args.config))
    if args.lang:
        pipeline.ocr_engine.lang = args.lang
    if args.model:
        pipeline.llm_client.config.model = args.model

    videos = collect_videos(input_path)
    if not videos:
        raise RuntimeError("No video files found")

    for video in tqdm(videos, desc="Processing videos"):
        pipeline.run(
            video_path=video,
            output_format=args.format,
            max_slides=args.max_slides,
            asr_language=args.lang,
        )


if __name__ == "__main__":
    main()
