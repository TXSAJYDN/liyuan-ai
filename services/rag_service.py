"""
RAG 问答服务：
- 整合知识库检索与千问模型生成
- 实现"RAG优先、通用生成兜底"的双路径机制
"""
import logging
from typing import List, Optional

from modules.knowledge_base import knowledge_base
from modules.qwen_model import qwen_model
from configs.settings import RAG_TOP_K, RAG_RELEVANCE_THRESHOLD

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 增强问答服务"""

    def __init__(self):
        pass

    def ensure_knowledge_base(self):
        if not knowledge_base.is_ready:
            loaded = knowledge_base.load()
            if not loaded:
                logger.info("知识库未找到，开始构建...")
                knowledge_base.build_index()

    def answer_question(self, question: str,
                        top_k: int = RAG_TOP_K,
                        threshold: float = RAG_RELEVANCE_THRESHOLD) -> dict:
        """回答戏曲相关问题"""
        self.ensure_knowledge_base()
        search_results = knowledge_base.search(question, top_k=top_k, threshold=threshold)
        if search_results:
            knowledge_context = "\n\n".join([
                f"[来源: {r.get('title', '未知')} ({r.get('genre', '')})] {r['text']}"
                for r in search_results
            ])
            logger.info(f"RAG 检索到 {len(search_results)} 条相关知识")
            if qwen_model.is_loaded:
                answer = qwen_model.answer_opera_question(
                    question, knowledge_context=knowledge_context, is_rag=True
                )
            else:
                answer = (
                    f"根据知识库检索到以下相关内容：\n\n{knowledge_context}\n\n"
                    "（注：千问模型尚未启动，暂时直接返回检索结果。启动模型后将生成更完整的回答。）"
                )
            return {
                "answer": answer,
                "source": "rag",
                "references": search_results,
                "note": "此答案基于戏曲专业知识库生成"
            }
        else:
            logger.info("知识库未命中，使用通用生成路径")
            if qwen_model.is_loaded:
                answer = qwen_model.answer_opera_question(question, is_rag=False)
            else:
                answer = (
                    "当前知识库中未检索到与此问题直接相关的内容，"
                    "且千问模型尚未启动。请启动模型后重试。"
                )
            return {
                "answer": answer,
                "source": "general",
                "references": [],
                "note": "此答案由大模型通用能力生成，仅供参考"
            }

    def analyze_video_with_rag(self, keyframe_paths: List[str],
                               visual_keywords: Optional[List[str]] = None) -> dict:
        """RAG 增强的视频结构化分析"""
        self.ensure_knowledge_base()
        knowledge_context = ""
        all_refs = []
        if visual_keywords:
            for keyword in visual_keywords:
                refs = knowledge_base.search(keyword, top_k=3)
                all_refs.extend(refs)
            if all_refs:
                seen_texts = set()
                unique_refs = []
                for ref in all_refs:
                    if ref["text"] not in seen_texts:
                        seen_texts.add(ref["text"])
                        unique_refs.append(ref)
                all_refs = unique_refs[:10]
                knowledge_context = "\n\n".join([r["text"] for r in all_refs])
        if qwen_model.is_loaded:
            analysis = qwen_model.analyze_opera_frames(
                keyframe_paths, knowledge_context=knowledge_context
            )
        else:
            analysis = (
                "千问模型尚未启动，无法进行多模态分析。\n"
                f"已准备 {len(keyframe_paths)} 个关键帧用于分析。\n"
            )
            if knowledge_context:
                analysis += f"检索到的相关知识：\n{knowledge_context}"
            else:
                analysis += "暂无检索到相关知识。请启动模型后重试。"
        return {
            "analysis": analysis,
            "knowledge_refs": all_refs,
        }


rag_service = RAGService()
