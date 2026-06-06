from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SETTINGS_DIR_ENV = "ANIMA_LORA_LAUNCHER_SETTINGS_DIR"
SETTINGS_FILE_ENV = "ANIMA_LORA_LAUNCHER_SETTINGS_FILE"
DEFAULT_SETTINGS_DIR = "user_settings"


def settings_path() -> Path:
    override = os.environ.get(SETTINGS_FILE_ENV)
    if override:
        return Path(override)

    settings_dir = os.environ.get(SETTINGS_DIR_ENV)
    if settings_dir:
        return Path(settings_dir) / "settings.json"

    app_root = Path(__file__).resolve().parents[2]
    return app_root / DEFAULT_SETTINGS_DIR / "settings.json"


def load_user_settings(path: Path | None = None) -> dict[str, Any]:
    target = path or settings_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_user_settings(data: dict[str, Any], path: Path | None = None) -> Path:
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target
