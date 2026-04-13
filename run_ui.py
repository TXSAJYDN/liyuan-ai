"""
启动自定义前端界面（FastAPI + Jinja2 模板）
原 Gradio 界面保留在 app/gradio_ui.py 可单独运行
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from configs.settings import GRADIO_PORT

if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=GRADIO_PORT,
        reload=True,
        log_level="info",
    )
