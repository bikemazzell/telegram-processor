from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


class SystemOps:
    """Wrap system interactions so workflow code can be unit-tested."""

    def which(self, tool: str) -> Optional[str]:
        return shutil.which(tool)

    def move(self, src: str, dst: str) -> None:
        shutil.move(src, dst)

    def run(self, *args, **kwargs):
        return subprocess.run(*args, **kwargs)

    def popen(self, *args, **kwargs):
        return subprocess.Popen(*args, **kwargs)

    def rmtree(self, path: Path) -> None:
        shutil.rmtree(path)

    def prompt(self, message: str) -> str:
        return input(message)
