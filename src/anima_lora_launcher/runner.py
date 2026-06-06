from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def build_training_command(sd_scripts_dir: Path, train_config: Path) -> list[str]:
    sd_scripts_dir = Path(sd_scripts_dir)
    train_config = Path(train_config)

    python_exe = sd_scripts_dir / "venv" / "Scripts" / "python.exe"
    script = sd_scripts_dir / "anima_train_network.py"

    if not python_exe.exists():
        python_exe = Path(sys.executable)

    return [
        str(python_exe),
        str(script),
        "--config_file",
        str(train_config),
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
