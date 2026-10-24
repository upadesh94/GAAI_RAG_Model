"""Root entrypoint launcher for DocChat AI Streamlit application.

Allows running either:
    streamlit run app.py
or
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_src = _root / "src"
for _p in (str(_src), str(_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.app import main

if __name__ == "__main__":
    main()
