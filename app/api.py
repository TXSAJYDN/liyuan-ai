"""
FastAPI 后端服务：
- 视频分析接口
- 关键帧语义检索接口
- 知识问答接口
- 任务状态接口
"""
import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import UPLOAD_DIR, KEYFRAME_DIR, OPERA_GENRES
from services.pipeline import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="梨园知见 - 融合视觉语义检索与知识增强生成的戏曲视频智能解析平台",
    description="基于多模态大模型的戏曲视频结构化分析、语义检索与专业问答",
    version="1.0.0"
)

APP_DIR = Path(__file__).parent.resolve()
app.mount("/keyframes", StaticFiles(directory=str(KEYFRAME_DIR)), name="keyframes")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

from app.web import router as web_router
app.include_router(web_router)


class SearchRequest(BaseModel):
    query: str
    video_name: Optional[str] = None
    top_k: int = 10

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

class AnalyzeOperaRequest(BaseModel):
    genre: str
    video_name: str


@app.get("/api/status")
async def get_status():
    return pipeline.get_status()

@app.get("/api/genres")
async def list_genres():
    return {"genres": OPERA_GENRES}

@app.get("/api/videos")
async def list_videos(genre: Optional[str] = None):
    try:
        videos = pipeline.list_available_videos(genre)
        return {"videos": videos, "total": len(videos)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件")
    allowed_exts = {".mp4", ".avi", ".mkv", ".flv", ".mov"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
    logger.info(f"视频上传成功: {save_path}")
    return {"filename": file.filename, "path": str(save_path), "size": len(content)}

@app.post("/api/analyze/upload")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件")
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
    try:
        result = pipeline.analyze_video(str(save_path))
        return result
    except Exception as e:
        logger.error(f"视频分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/opera")
async def analyze_opera_video(request: AnalyzeOperaRequest):
    try:
        process_result = pipeline.process_opera_video(request.genre, request.video_name)
        keyframe_paths = [kf["path"] for kf in process_result["keyframes"]]
        analysis_result = pipeline.analyze_video(
            process_result["video_path"], keyframe_paths=keyframe_paths
        )
        return {
            "video_info": process_result["video_info"],
            "segments_count": len(process_result["segments"]),
            **analysis_result
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"视频分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def semantic_search(request: SearchRequest):
    try:
        results = pipeline.semantic_search(request.query, video_name=request.video_name)
        return {"query": request.query, "results": results, "total": len(results)}
    except Exception as e:
        logger.error(f"语义检索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/qa")
async def question_answer(request: QuestionRequest):
    try:
        result = pipeline.ask_question(request.question)
        return result
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/init/knowledge_base")
async def init_knowledge_base():
    try:
        pipeline.init_knowledge_base()
        from modules.knowledge_base import knowledge_base as kb
        return {"status": "ok", "docs_count": len(kb.documents) if kb.documents else 0}
    except Exception as e:
        logger.error(f"知识库初始化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/init/qwen_model")
async def init_qwen_model():
    try:
        pipeline.load_qwen_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        logger.error(f"千问模型加载失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/keyframe/{video_name}/{filename}")
async def get_keyframe(video_name: str, filename: str):
    file_path = KEYFRAME_DIR / video_name / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="关键帧不存在")
    return FileResponse(str(file_path))
