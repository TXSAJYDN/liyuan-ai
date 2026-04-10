"""
梨园AI 全局配置文件
"""
import os
from pathlib import Path

# ============ 路径配置 ============
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 原始戏曲数据目录
OPERA_DATA_DIR = Path("/srv/nas_data/opera")

# 千问多模态模型路径（配置好，按需加载）
QWEN_MODEL_PATH = "/home/shq/data/models/Qwen3-omni-30B-A3B-Instruct"

# 本地数据存储目录
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
KEYFRAME_DIR = DATA_DIR / "keyframes"
CACHE_DIR = DATA_DIR / "cache"

# 知识库目录
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
VECTOR_DB_DIR = KNOWLEDGE_BASE_DIR / "vector_store"
RAW_KNOWLEDGE_DIR = KNOWLEDGE_BASE_DIR / "raw_texts"

# ============ 戏曲数据子目录映射 ============
OPERA_GENRES = {
    "bangzi": "梆子",
    "em": "二人台",
    "gaoqiang": "高腔",
    "jiangnan": "江南丝竹",
    "lantern": "花灯戏",
    "model": "样板戏",
    "other": "其他",
    "region": "地方戏",
}

# ============ 视频处理配置 ============
VIDEO_SEGMENT_DURATION = 60
KEYFRAME_INTERVAL = 2
SCENE_CHANGE_THRESHOLD = 30.0
KEYFRAME_SIZE = (640, 480)

# ============ CLIP 模型配置 ============
CLIP_MODEL_NAME = "ViT-B-16"
CLIP_PRETRAINED = "openai"

# ============ 千问模型配置 ============
QWEN_CONFIG = {
    "model_path": QWEN_MODEL_PATH,
    "device": "auto",
    "torch_dtype": "auto",
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
}

# ============ RAG 配置 ============
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
RAG_TOP_K = 5
RAG_RELEVANCE_THRESHOLD = 0.5

# ============ 语义检索配置 ============
CLIP_TOP_K = 10
CLIP_SIMILARITY_THRESHOLD = 0.2

# ============ 视频分析配置 ============
MAX_ANALYSIS_FRAMES = 100  # 用于结构化分析的最大关键帧数量（均匀采样）

# ============ 服务配置 ============
API_HOST = "0.0.0.0"
API_PORT = 8000
GRADIO_PORT = 7860

# ============ 确保目录存在 ============
def ensure_dirs():
    for d in [UPLOAD_DIR, PROCESSED_DIR, KEYFRAME_DIR, CACHE_DIR,
              VECTOR_DB_DIR, RAW_KNOWLEDGE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

ensure_dirs()
