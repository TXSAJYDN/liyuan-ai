"""
启动 Gradio 前端界面
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from app.gradio_ui import main

if __name__ == "__main__":
    main()
