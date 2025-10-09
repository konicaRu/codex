from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from .env_check import collect_environment_status, load_config
from .pipeline import Pipeline

APP = Flask(__name__)
APP.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB uploads
APP.config["DEFAULT_CONFIG_PATH"] = str(Path("config.yaml").resolve())

_PIPELINE_CACHE: Dict[Path, Pipeline] = {}
_PIPELINE_LOCK = threading.Lock()


@APP.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def get_pipeline(config_path: Path) -> Pipeline:
    with _PIPELINE_LOCK:
        pipeline = _PIPELINE_CACHE.get(config_path)
        if pipeline is None:
            pipeline = Pipeline(config_path)
            _PIPELINE_CACHE[config_path] = pipeline
        return pipeline


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:  # noqa: BLE001
        raise ValueError("Параметр maxSlides должен быть числом") from exc


def _resolve_config_path(value: Optional[str]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return Path(APP.config.get("DEFAULT_CONFIG_PATH", "config.yaml")).resolve()


@APP.route("/api/status", methods=["GET", "OPTIONS"])
def status():
    if request.method == "OPTIONS":
        return ("", 204)

    config_path = _resolve_config_path(request.args.get("config"))
    try:
        config = load_config(config_path)
        env_status = collect_environment_status(config_path)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "message": str(exc)}), 400

    llm_cfg = config.get("llm", {})
    overall_ok = all(item.get("ok", False) for item in env_status.values())
    response = {
        "ok": overall_ok,
        "config": {
            "model": llm_cfg.get("model", ""),
            "api_base": llm_cfg.get("api_base", "http://localhost:1234/v1"),
        },
        "environment": env_status,
    }
    return jsonify(response)


@APP.route("/api/process", methods=["POST", "OPTIONS"])
def process_video():
    if request.method == "OPTIONS":
        return ("", 204)

    config_path = _resolve_config_path(request.form.get("config"))
    if not config_path.exists():
        return jsonify({"ok": False, "message": f"config.yaml не найден: {config_path}"}), 400

    if "file" not in request.files:
        return jsonify({"ok": False, "message": "Загрузите видеофайл"}), 400

    file_storage = request.files["file"]
    if file_storage.filename == "":
        return jsonify({"ok": False, "message": "Файл не выбран"}), 400

    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file_storage.filename)
    video_path = upload_dir / filename
    file_storage.save(video_path)

    output_format = request.form.get("format", "pdf")
    if output_format not in {"pdf", "pptx"}:
        return jsonify({"ok": False, "message": "format должен быть pdf или pptx"}), 400

    max_slides = None
    try:
        max_slides = _parse_int(request.form.get("maxSlides"))
    except ValueError as exc:  # noqa: BLE001
        return jsonify({"ok": False, "message": str(exc)}), 400

    language = request.form.get("lang")
    model_override = request.form.get("model")

    try:
        pipeline = get_pipeline(config_path)
        if language:
            pipeline.ocr_engine.lang = language
        if model_override:
            pipeline.llm_client.config.model = model_override

        result = pipeline.run(
            video_path=video_path,
            output_format=output_format,
            max_slides=max_slides,
            asr_language=language,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "message": str(exc)}), 500
    finally:
        try:
            video_path.unlink()
        except Exception:  # noqa: BLE001
            pass

    report_path = result.output_path
    log_path = result.log_path

    response = {
        "ok": True,
        "report_path": str(report_path) if report_path else "",
        "log_path": str(log_path) if log_path else "",
        "report_url": f"/api/reports/{report_path.name}" if report_path else "",
        "log_url": f"/api/logs/{log_path.name}" if log_path else "",
    }
    return jsonify(response)


@APP.route("/api/reports/<path:filename>")
def download_report(filename: str):
    reports_dir = Path("reports")
    if not (reports_dir / filename).exists():
        return jsonify({"ok": False, "message": "Файл не найден"}), 404
    return send_from_directory(reports_dir, filename, as_attachment=True)


@APP.route("/api/logs/<path:filename>")
def download_log(filename: str):
    logs_dir = Path("logs")
    if not (logs_dir / filename).exists():
        return jsonify({"ok": False, "message": "Файл не найден"}), 404
    return send_from_directory(logs_dir, filename, as_attachment=True)


@APP.route("/")
def index():
    html_path = Path(__file__).resolve().parent / "gui" / "index.html"
    if not html_path.exists():
        return "GUI не найден", 404
    return (
        html_path.read_text(encoding="utf-8"),
        200,
        {"Content-Type": "text/html; charset=utf-8"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Веб-интерфейс для конвейера отчётов")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    APP.config["DEFAULT_CONFIG_PATH"] = str(Path(args.config).resolve())
    APP.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
