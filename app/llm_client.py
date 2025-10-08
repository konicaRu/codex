from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import requests

from .models import ActionsResult, MinutesResult, SegmentLLMResult, TopicsResult, VideoSegment


@dataclass
class LLMConfig:
    api_base: str
    model: str
    temperature: float = 0.2
    max_retries: int = 3
    summary_prompt: str = ""
    cluster_prompt: str = ""
    action_prompt: str = ""
    minutes_prompt: str = ""


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _chat(self, prompt: str, input_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": prompt},
            ],
        }
        if input_json is not None:
            payload["messages"].append(
                {"role": "user", "content": json.dumps(input_json, ensure_ascii=False)}
            )
        else:
            payload["messages"].append({"role": "user", "content": ""})

        for attempt in range(self.config.max_retries):
            response = requests.post(
                f"{self.config.api_base}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=300,
            )
            if response.status_code == 200:
                data = response.json()
                try:
                    content = data["choices"][0]["message"]["content"]
                    return json.loads(content)
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Failed to parse LLM response: {data}") from exc
            if attempt == self.config.max_retries - 1:
                response.raise_for_status()
        return {}

    def summarize_segment(
        self, segment: VideoSegment, context: Optional[Dict[str, Any]] = None
    ) -> SegmentLLMResult:
        payload = {
            "ocr_text": segment.ocr_text,
            "asr_text": segment.asr_text,
            "context": context or {},
        }
        result = self._chat(self.config.summary_prompt, payload)
        return SegmentLLMResult(
            title=result.get("title", ""),
            bullets=result.get("bullets", []),
        )

    def cluster_topics(self, segments: Iterable[VideoSegment]) -> TopicsResult:
        payload = [
            {"index": seg.index, "title": seg.summary_title, "bullets": seg.summary_bullets}
            for seg in segments
        ]
        result = self._chat(self.config.cluster_prompt, {"segments": payload})
        return TopicsResult(
            topics=result.get("topics", []),
            summary=result.get("summary", ""),
        )

    def make_action_items(self, segments: Iterable[VideoSegment]) -> ActionsResult:
        payload = [
            {"index": seg.index, "title": seg.summary_title, "bullets": seg.summary_bullets}
            for seg in segments
        ]
        result = self._chat(self.config.action_prompt, {"segments": payload})
        return ActionsResult(actions=result.get("actions", []))

    def make_global_minutes(self, segments: Iterable[VideoSegment]) -> MinutesResult:
        payload = [
            {"index": seg.index, "title": seg.summary_title, "bullets": seg.summary_bullets}
            for seg in segments
        ]
        result = self._chat(self.config.minutes_prompt, {"segments": payload})
        return MinutesResult(bullets=result.get("bullets", []))


def build_client(config: Dict[str, Any]) -> LLMClient:
    api_base = (
        config.get("api_base")
        or os.getenv("OPENAI_API_BASE")
        or "http://localhost:1234/v1"
    )
    llm_config = LLMConfig(
        api_base=api_base.rstrip("/"),
        model=config.get("model", ""),
        temperature=config.get("temperature", 0.2),
        max_retries=config.get("max_retries", 3),
        summary_prompt=config.get("summary_prompt", ""),
        cluster_prompt=config.get("cluster_prompt", ""),
        action_prompt=config.get("action_prompt", ""),
        minutes_prompt=config.get("minutes_prompt", ""),
    )
    return LLMClient(llm_config)


__all__ = ["LLMClient", "LLMConfig", "build_client"]
