"""
页面路由：返回 Jinja2 HTML 模板
与 api.py 的 JSON 接口共存于同一个 FastAPI 实例
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

router = APIRouter()


@router.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.get("/analyze")
async def analyze(request: Request):
    return templates.TemplateResponse(request, "analyze.html")


@router.get("/search")
async def search(request: Request):
    return templates.TemplateResponse(request, "search.html")


@router.get("/qa")
async def qa(request: Request):
    return templates.TemplateResponse(request, "qa.html")
