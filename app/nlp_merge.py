from __future__ import annotations

import re
from typing import Tuple


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def merge_ocr_asr(ocr_text: str, asr_text: str) -> Tuple[str, str]:
    ocr_norm = normalize_text(ocr_text)
    asr_norm = normalize_text(asr_text)

    title = ocr_norm.split(". ")[0][:120] if ocr_norm else asr_norm.split(". ")[0][:120]
    if not title:
        title = asr_norm[:120]

    if ocr_norm and asr_norm:
        if ocr_norm.lower() in asr_norm.lower():
            combined = asr_norm
        else:
            combined = f"{ocr_norm}\n{asr_norm}"
    else:
        combined = ocr_norm or asr_norm

    return title.strip(), combined.strip()


__all__ = ["merge_ocr_asr", "normalize_text"]
