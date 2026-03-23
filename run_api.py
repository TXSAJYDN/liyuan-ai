"""
启动 FastAPI 后端服务
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from configs.settings import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
