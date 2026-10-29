"""Root entrypoint for Streamlit cloud deployments (Render, Streamlit Cloud, Hugging Face)."""
import os
import sys
from pathlib import Path
import runpy

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__" or "streamlit" in sys.modules:
    runpy.run_path(str(PROJECT_ROOT / "app" / "streamlit_app.py"), run_name="__main__")
