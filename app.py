"""Streamlit entry point for the Spam Email Classifier.

Run with::

    streamlit run app.py

The application itself lives in :mod:`src.ui`; this module only guarantees that
the project root is importable no matter which directory Streamlit is launched
from.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui import main  # noqa: E402  (import after sys.path setup)

if __name__ == "__main__":
    main()
