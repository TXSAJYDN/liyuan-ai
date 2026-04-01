"""
核心流水线服务：
- 串联视频处理、知识库检索、CLIP检索、千问分析等模块
- 提供完整的业务逻辑编排
"""
import logging
import shutil
from pathlib import Path
from typing import Optional, List

from modules.video_processor import video_processor
from modules.knowledge_base import knowledge_base
from modules.clip_retriever import clip_retriever
from modules.qwen_model import qwen_model
from services.rag_service import rag_service
from configs.settings import UPLOAD_DIR, OPERA_DATA_DIR, MAX_ANALYSIS_FRAMES

logger = logging.getLogger(__name__)


class Pipeline:
    """梨园AI 核心业务流水线"""

    def __init__(self):
        self._kb_initialized = False

    def init_knowledge_base(self):
        if self._kb_initialized and knowledge_base.is_ready:
            return
        loaded = knowledge_base.load()
        if not loaded:
            logger.info("知识库不存在，开始构建...")
            knowledge_base.build_index()
        self._kb_initialized = True

    def load_qwen_model(self):
        if not qwen_model.is_loaded:
            qwen_model.load_model()

    def process_uploaded_video(self, video_path: str) -> dict:
        result = video_processor.process_video(video_path)
        if result["keyframes"]:
            clip_retriever.build_index(result["keyframes"])
            video_name = Path(video_path).stem
            clip_retriever.save_index(video_name)
            result["clip_index_ready"] = True
        else:
            result["clip_index_ready"] = False
        return result

    def process_opera_video(self, genre: str, video_name: str) -> dict:
        video_dir = OPERA_DATA_DIR / genre
        video_file = None
        for ext in (".mp4", ".avi", ".mkv", ".flv", ".mov"):
            candidate = video_dir / f"{video_name}{ext}"
            if candidate.exists():
                video_file = candidate
                break
        if video_file is None:
            raise FileNotFoundError(f"未找到视频: {genre}/{video_name}")
        return self.process_uploaded_video(str(video_file))

    def analyze_video(self, video_path: str,
                      keyframe_paths: Optional[List[str]] = None) -> dict:
        self.init_knowledge_base()
        if keyframe_paths is None:
            result = self.process_uploaded_video(video_path)
            keyframe_paths = [kf["path"] for kf in result["keyframes"]]
        # 均匀采样关键帧，覆盖整个视频
        if len(keyframe_paths) <= MAX_ANALYSIS_FRAMES:
            sample_frames = keyframe_paths
        else:
            import numpy as np
            indices = np.linspace(0, len(keyframe_paths) - 1, MAX_ANALYSIS_FRAMES, dtype=int)
            sample_frames = [keyframe_paths[i] for i in indices]
        analysis_result = rag_service.analyze_video_with_rag(
            sample_frames,
            visual_keywords=["戏曲表演", "行当", "动作程式", "服饰"]
        )
        return {
            "keyframe_count": len(keyframe_paths),
            "sampled_frames": len(sample_frames),
            "analysis": analysis_result["analysis"],
            "knowledge_refs": analysis_result["knowledge_refs"],
        }

    def semantic_search(self, query_text: str, video_name: Optional[str] = None) -> List[dict]:
        if video_name and not clip_retriever.is_ready:
            clip_retriever.load_index(video_name)
        if not clip_retriever.is_ready:
            return []
        results = clip_retriever.search(query_text)
        if results and qwen_model.is_loaded:
            top_result = results[0]
            explanation = qwen_model.explain_keyframe_match(
                top_result["path"], query_text
            )
            results[0]["explanation"] = explanation
        return results

    def ask_question(self, question: str) -> dict:
        self.init_knowledge_base()
        return rag_service.answer_question(question)

    def list_available_videos(self, genre: Optional[str] = None) -> List[dict]:
        return video_processor.list_opera_videos(genre)

    def get_status(self) -> dict:
        return {
            "knowledge_base_ready": knowledge_base.is_ready,
            "clip_index_ready": clip_retriever.is_ready,
            "qwen_model_loaded": qwen_model.is_loaded,
            "knowledge_base_docs": len(knowledge_base.documents) if knowledge_base.documents else 0,
            "clip_index_size": clip_retriever.index.ntotal if clip_retriever.index else 0,
        }


pipeline = Pipeline()
