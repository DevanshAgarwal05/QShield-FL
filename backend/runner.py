from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def run_main_py(project_root: Path, log_file: Optional[Path] = None) -> int:
    """
    Runs main.py as a separate process.
    Safe because it does not kill/interrupt any existing training.
    """
    main_path = project_root / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"main.py not found at: {main_path}")

    cmd = ["python", "-u", str(main_path)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, cwd=str(project_root), stdout=f, stderr=subprocess.STDOUT)
            return p.wait()

    p = subprocess.Popen(cmd, cwd=str(project_root))
    return p.wait()
