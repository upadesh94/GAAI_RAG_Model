#!/usr/bin/env python3
"""Root entrypoint launcher for DocChat CLI.

Allows running either:
    python cli.py <pdf_path>
or
    python src/cli.py <pdf_path>
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_src = _root / "src"
for _p in (str(_src), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.cli import main

if __name__ == "__main__":
    main()
