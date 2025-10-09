from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
from skimage.metrics import structural_similarity as ssim


@dataclass
class SceneSegment:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    best_frame_path: Path


class VideoSceneSplitter:
    """Split video into segments using histogram difference and SSIM."""

    def __init__(
        self,
        output_dir: Path,
        histogram_threshold: float = 0.35,
        ssim_threshold: float = 0.65,
        min_segment_duration: float = 4.0,
        max_segment_duration: float = 180.0,
        frame_sample_rate: float = 1.0,
        max_slides: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.histogram_threshold = histogram_threshold
        self.ssim_threshold = ssim_threshold
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.frame_sample_rate = frame_sample_rate
        self.max_slides = max_slides
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, video_path: Path) -> List[SceneSegment]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count else 0
        interval_frames = max(int(fps * self.frame_sample_rate), 1)

        segments: List[SceneSegment] = []
        current_start_frame = 0
        current_best_frame = None
        current_best_score = -math.inf
        current_start_time = 0.0
        last_saved_frame = None
        last_saved_image = None

        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % interval_frames != 0:
                frame_index += 1
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if last_saved_frame is not None:
                diff = cv2.compareHist(last_saved_frame, hist, cv2.HISTCMP_BHATTACHARYYA)
                gray_prev = cv2.cvtColor(last_saved_image, cv2.COLOR_BGR2GRAY)
                gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_ssim, _ = ssim(gray_prev, gray_curr, full=True)
            else:
                diff = 0.0
                frame_ssim = 1.0

            current_time = frame_index / fps

            variance = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if variance > current_best_score:
                current_best_score = variance
                best_frame_file = self.output_dir / f"segment_{len(segments):03d}_frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(best_frame_file), frame)
                current_best_frame = best_frame_file

            boundary = self._should_split(
                diff,
                frame_ssim,
                current_time - current_start_time,
                duration,
                len(segments),
            )

            if boundary and current_best_frame is not None:
                end_time = current_time
                segments.append(
                    SceneSegment(
                        start_frame=current_start_frame,
                        end_frame=frame_index,
                        start_time=current_start_time,
                        end_time=end_time,
                        best_frame_path=current_best_frame,
                    )
                )
                current_start_frame = frame_index
                current_start_time = current_time
                current_best_score = -math.inf
                current_best_frame = None

            last_saved_frame = hist
            last_saved_image = frame
            frame_index += 1

        # finalize last segment
        if current_best_frame is not None:
            segments.append(
                SceneSegment(
                    start_frame=current_start_frame,
                    end_frame=frame_index,
                    start_time=current_start_time,
                    end_time=duration or (frame_index / fps),
                    best_frame_path=current_best_frame,
                )
            )

        capture.release()

        merged = self._merge_short_segments(segments)
        if self.max_slides is not None:
            merged = merged[: self.max_slides]
        return merged

    def _should_split(
        self,
        histogram_diff: float,
        frame_ssim: float,
        segment_duration: float,
        total_duration: float,
        segments_count: int,
    ) -> bool:
        if segment_duration >= self.max_segment_duration:
            return True
        if segment_duration < self.min_segment_duration:
            return False
        if histogram_diff > self.histogram_threshold and frame_ssim < self.ssim_threshold:
            return True
        if self.max_slides is not None and segments_count + 1 >= self.max_slides:
            return False
        return False

    def _merge_short_segments(self, segments: Iterable[SceneSegment]) -> List[SceneSegment]:
        merged: List[SceneSegment] = []
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue
            if seg.end_time - seg.start_time < self.min_segment_duration:
                prev = merged[-1]
                merged[-1] = SceneSegment(
                    start_frame=prev.start_frame,
                    end_frame=seg.end_frame,
                    start_time=prev.start_time,
                    end_time=seg.end_time,
                    best_frame_path=prev.best_frame_path,
                )
            else:
                merged.append(seg)
        return merged


__all__ = ["SceneSegment", "VideoSceneSplitter"]
