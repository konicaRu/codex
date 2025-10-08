from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict

import requests
import yaml

from .llm_client import build_client


def _check_binary(name: str, configured_path: str | None, install_hint: str) -> Dict[str, Any]:
    if configured_path:
        path = Path(configured_path)
        if path.exists():
            return {
                "ok": True,
                "path": str(path),
                "message": f"Используется путь из config.yaml: {path}",
            }
    found = shutil.which(configured_path or name)
    if found:
        return {"ok": True, "path": found, "message": f"Найден в PATH: {found}"}
    return {
        "ok": False,
        "message": install_hint,
    }


def _check_llm_server(api_base: str) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/models"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return {
                "ok": True,
                "message": "LM Studio отвечает на /models",
            }
        return {
            "ok": False,
            "message": f"Сервер ответил кодом {response.status_code}. Проверьте, что в LM Studio включён OpenAI API.",
        }
    except requests.RequestException as exc:  # noqa: PERF203
        return {
            "ok": False,
            "message": f"Не удалось подключиться к {api_base}: {exc}",
        }


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def collect_environment_status(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Не найден config.yaml по пути {config_path}")

    config = load_config(config_path)
    paths = config.get("paths", {})
    llm_cfg = config.get("llm", {})

    status: Dict[str, Any] = {
        "ffmpeg": _check_binary(
            "ffmpeg",
            paths.get("ffmpeg"),
            "ffmpeg не найден. Установите ffmpeg и добавьте его в PATH или пропишите путь в config.yaml → paths.ffmpeg.",
        ),
        "tesseract": _check_binary(
            "tesseract",
            paths.get("tesseract"),
            "Tesseract не найден. Установите Tesseract OCR с языками rus и eng, затем обновите PATH или config.yaml → paths.tesseract.",
        ),
    }

    api_base = (
        llm_cfg.get("api_base")
        or os.getenv("OPENAI_API_BASE")
        or "http://localhost:1234/v1"
    )
    status["lm_studio"] = _check_llm_server(api_base)

    try:
        client = build_client(llm_cfg)
        model = client.config.model
    except Exception as exc:  # noqa: BLE001
        status["model"] = {
            "ok": False,
            "message": f"Ошибка разбора настроек LLM: {exc}",
        }
    else:
        if model:
            status["model"] = {
                "ok": True,
                "message": f"Текущая модель: {model}",
            }
        else:
            status["model"] = {
                "ok": False,
                "message": "В config.yaml → llm.model не указано имя модели. Укажите модель, загруженную в LM Studio.",
            }

    return status


__all__ = ["collect_environment_status", "load_config"]
