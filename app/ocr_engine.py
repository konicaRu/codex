from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import pytesseract


class OCREngine:
    def __init__(
        self,
        lang: str = "rus+eng",
        tesseract_cmd: Optional[str] = None,
        psm: int = 6,
        oem: int = 3,
    ) -> None:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = lang
        self.psm = psm
        self.oem = oem

    def extract_text(self, image_path: Path) -> str:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = f"--oem {self.oem} --psm {self.psm}"
        text = pytesseract.image_to_string(thresh, lang=self.lang, config=config)
        return text.strip()


__all__ = ["OCREngine"]
