from __future__ import annotations

from pathlib import Path
from typing import Iterable

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .models import ActionsResult, MinutesResult, TopicsResult, VideoSegment


class PDFRenderer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        video_name: str,
        segments: Iterable[VideoSegment],
        topics: TopicsResult | None,
        actions: ActionsResult | None,
        minutes: MinutesResult | None,
    ) -> Path:
        output_path = self.output_dir / f"{video_name}_report.pdf"
        pdf = canvas.Canvas(str(output_path), pagesize=A4)
        width, height = A4

        for segment in segments:
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(40, height - 60, segment.summary_title[:90] or "Слайд")
            pdf.setFont("Helvetica", 12)
            bullet_y = height - 90
            for bullet in segment.summary_bullets[:3]:
                pdf.drawString(60, bullet_y, f"• {bullet[:130]}")
                bullet_y -= 20

            if segment.best_frame_path.exists():
                image = ImageReader(str(segment.best_frame_path))
                img_width, img_height = image.getSize()
                ratio = min(400 / img_width, 300 / img_height)
                pdf.drawImage(
                    image,
                    40,
                    bullet_y - 320,
                    width=img_width * ratio,
                    height=img_height * ratio,
                    preserveAspectRatio=True,
                )
            pdf.showPage()

        self._render_summary(pdf, topics, actions, minutes, width, height)
        pdf.save()
        return output_path

    def _render_summary(
        self,
        pdf: canvas.Canvas,
        topics: TopicsResult | None,
        actions: ActionsResult | None,
        minutes: MinutesResult | None,
        width: float,
        height: float,
    ) -> None:
        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawString(40, height - 60, "Итоги встречи")

        y = height - 100
        if topics:
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(40, y, "Темы:")
            y -= 20
            pdf.setFont("Helvetica", 12)
            for topic in topics.topics:
                pdf.drawString(60, y, f"• {topic[:130]}")
                y -= 18
            if topics.summary:
                pdf.drawString(40, y, f"Резюме: {topics.summary[:150]}")
                y -= 24

        if actions and actions.actions:
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(40, y, "Экшен-пункты:")
            y -= 20
            pdf.setFont("Helvetica", 12)
            for action in actions.actions:
                pdf.drawString(60, y, f"• {action[:130]}")
                y -= 18

        if minutes and minutes.bullets:
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(40, y, "Главные выводы:")
            y -= 20
            pdf.setFont("Helvetica", 12)
            for bullet in minutes.bullets:
                pdf.drawString(60, y, f"• {bullet[:130]}")
                y -= 18

        pdf.showPage()


__all__ = ["PDFRenderer"]
