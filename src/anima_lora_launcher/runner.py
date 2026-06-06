from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def sd_scripts_python(sd_scripts_dir: Path) -> Path:
    python_exe = Path(sd_scripts_dir) / "venv" / "Scripts" / "python.exe"
    if python_exe.exists():
        return python_exe
    return Path(sys.executable)


def build_training_command(sd_scripts_dir: Path, train_config: Path) -> list[str]:
    sd_scripts_dir = Path(sd_scripts_dir)
    train_config = Path(train_config)

    script = sd_scripts_dir / "anima_train_network.py"

    return [
        str(sd_scripts_python(sd_scripts_dir)),
        str(script),
        "--config_file",
        str(train_config),
    ]


def find_wd14_tagger_script(sd_scripts_dir: Path) -> Path | None:
    sd_scripts_dir = Path(sd_scripts_dir)
    candidates = [
        sd_scripts_dir / "finetune" / "tag_images_by_wd14_tagger.py",
        sd_scripts_dir / "tag_images_by_wd14_tagger.py",
    ]
    for script in candidates:
        if script.exists():
            return script
    return None


def build_wd14_command(
    sd_scripts_dir: Path,
    image_dir: Path,
    *,
    caption_extension: str = ".txt",
    batch_size: int = 4,
    repo_id: str = "SmilingWolf/wd-swinv2-tagger-v3",
) -> list[str]:
    script = find_wd14_tagger_script(sd_scripts_dir)
    if script is None:
        raise FileNotFoundError("tag_images_by_wd14_tagger.py was not found in sd-scripts.")

    return [
        str(sd_scripts_python(sd_scripts_dir)),
        str(script),
        "--onnx",
        "--repo_id",
        repo_id,
        "--batch_size",
        str(max(1, int(batch_size))),
        "--caption_extension",
        caption_extension,
        str(image_dir),
    ]


def popen_training(
    *,
    sd_scripts_dir: Path,
    command: list[str],
) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    return subprocess.Popen(
        command,
        cwd=str(sd_scripts_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=1,
        creationflags=creationflags,
    )


def decode_process_output(raw: bytes) -> str:
    for encoding in ("utf-8", "cp932", "mbcs"):
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            pass
    return raw.decode("utf-8", errors="replace")


def stop_process_tree(process: subprocess.Popen[bytes]) -> tuple[bool, str]:
    pid = process.pid
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
        )
        return result.returncode == 0, decode_process_output(result.stdout).strip()

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    return True, f"terminated pid {pid}"
