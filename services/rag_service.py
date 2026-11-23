"""
RAG 问答服务：
- 整合知识库混合检索与千问模型生成
- 轻量查询改写、跨查询结果合并、近似去重与上下文预算
- 为每条参考资料分配 [S#] 引用编号，降低无依据生成
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

from configs.settings import (
    RAG_TOP_K,
    RAG_RELEVANCE_THRESHOLD,
    RAG_MAX_CONTEXT_CHARS,
)
from modules.knowledge_base import knowledge_base
from modules.qwen_model import qwen_model

logger = logging.getLogger(__name__)

_QUERY_PREFIXES = (
    "请问", "请", "麻烦", "我想知道", "我想了解", "能否", "可以",
)
_QUERY_NOISE_PHRASES = (
    "有什么区别", "有什么不同", "是什么意思", "什么是", "是什么",
    "有哪些", "为什么", "怎么样", "怎么", "如何", "介绍一下", "介绍",
)
_WHITESPACE_RE = re.compile(r"\s+")


class RAGService:
    """RAG 增强问答服务"""

    def ensure_knowledge_base(self):
        if not knowledge_base.is_ready:
            loaded = knowledge_base.load()
            if not loaded:
                logger.info("知识库未找到，开始构建...")
                knowledge_base.build_index()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return _WHITESPACE_RE.sub(" ", (text or "").lower()).strip()

    def _query_variants(self, question: str) -> List[str]:
        """生成轻量查询变体，不调用额外 LLM。"""
        question = (question or "").strip()
        if not question:
            return []

        cleaned = question
        for prefix in _QUERY_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        for phrase in _QUERY_NOISE_PHRASES:
            cleaned = cleaned.replace(phrase, " ")
        cleaned = re.sub(r"[？?！!。，,；;：:\s]+", " ", cleaned)
        cleaned = " ".join(cleaned.split()).strip()

        variants = [question]
        if cleaned and len(cleaned) >= 2 and cleaned != question:
            variants.append(cleaned)
        return list(dict.fromkeys(variants))[:2]

    def _merge_results(self, results: List[dict]) -> List[dict]:
        """合并多条查询的检索结果，按规范化正文去重并保留更高分。"""
        merged: Dict[str, dict] = {}
        for result in results:
            text = (result.get("text") or "").strip()
            if not text:
                continue
            key = self._normalize_text(text)
            previous = merged.get(key)
            if previous is None or result.get("score", 0.0) > previous.get("score", 0.0):
                merged[key] = result
        return sorted(
            merged.values(),
            key=lambda item: (
                item.get("score", 0.0),
                item.get("dense_score", 0.0),
                item.get("bm25_score", 0.0),
            ),
            reverse=True,
        )

    def retrieve(self, question: str, top_k: int = RAG_TOP_K,
                 threshold: float = RAG_RELEVANCE_THRESHOLD) -> List[dict]:
        """多查询检索并合并结果。"""
        if int(top_k) <= 0:
            return []
        self.ensure_knowledge_base()
        variants = self._query_variants(question)
        if not variants:
            return []

        all_results: List[dict] = []
        per_query_k = max(int(top_k), RAG_TOP_K)
        for variant in variants:
            try:
                all_results.extend(knowledge_base.search(
                    variant, top_k=per_query_k, threshold=threshold
                ))
            except Exception as exc:
                logger.error(f"检索查询「{variant}」失败: {exc}", exc_info=True)

        merged = self._merge_results(all_results)
        # 多留一些候选给上下文阶段做近似去重和预算裁剪。
        return merged[:max(int(top_k) * 2, int(top_k) + 3)]

    @staticmethod
    def _truncate_at_boundary(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        cut = text[:max(0, limit - 1)]
        last_boundary = -1
        for separator in ("\n", "。", "！", "？", "；", ". ", "! ", "? ", "；", "，", ", ", " "):
            position = cut.rfind(separator)
            if position >= int(limit * 0.5):
                last_boundary = max(last_boundary, position + len(separator))
        if last_boundary > 0:
            cut = cut[:last_boundary]
        return cut.rstrip() + "…"

    def _format_context(self, results: List[dict],
                        limit: Optional[int] = None) -> Tuple[str, List[dict]]:
        """格式化带 [S#] 编号的上下文，并执行近似去重与总字符预算。"""
        parts: List[str] = []
        included: List[dict] = []
        remaining = int(RAG_MAX_CONTEXT_CHARS)

        for result in results:
            if limit is not None and len(included) >= max(0, int(limit)):
                break
            text = (result.get("text") or "").strip()
            if not text:
                continue
            if any(
                knowledge_base._is_near_duplicate(text, ref.get("text", ""))
                for ref in included
            ):
                continue

            citation_id = f"S{len(included) + 1}"
            title = result.get("title") or "未知"
            genre = result.get("genre") or "未分类"
            header = f"[{citation_id}] 标题：{title}｜剧种：{genre}"
            overhead = len(header) + 1
            if remaining <= overhead + 40:
                break

            body = self._truncate_at_boundary(text, remaining - overhead)
            if not body:
                break
            part = f"{header}\n{body}"
            if len(part) > remaining:
                body = self._truncate_at_boundary(text, max(0, remaining - overhead))
                part = f"{header}\n{body}"
            if not body or len(part) > remaining:
                break

            parts.append(part)
            reference = dict(result)
            reference["citation_id"] = citation_id
            included.append(reference)
            remaining -= len(part) + 2
            if remaining <= 0:
                break

        return "\n\n".join(parts), included

    def prepare_answer(self, question: str,
                       top_k: int = RAG_TOP_K,
                       threshold: float = RAG_RELEVANCE_THRESHOLD) -> dict:
        """完成检索和提示词构造，供阻塞式 API 与流式 Gradio 共用。"""
        results = self.retrieve(question, top_k=top_k, threshold=threshold)
        context, references = self._format_context(
            results, limit=max(1, int(top_k))
        )
        is_rag = bool(references)

        system_prompt = qwen_model.opera_qa_system_prompt()
        prompt = qwen_model.build_opera_qa_prompt(
            question,
            knowledge_context=context,
            is_rag=is_rag,
        )

        if is_rag:
            fallback_answer = (
                "根据知识库检索到以下相关内容：\n\n"
                f"{context}\n\n"
                "（注：千问模型尚未启动，暂时直接返回检索结果。启动模型后将生成更完整的回答。）"
            )
        else:
            fallback_answer = (
                "当前知识库中未检索到与此问题直接相关的内容，"
                "且千问模型尚未启动。请启动模型后重试。"
            )

        return {
            "question": question,
            "results": results,
            "context": context,
            "references": references,
            "is_rag": is_rag,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "fallback_answer": fallback_answer,
        }

    def answer_question(self, question: str,
                        top_k: int = RAG_TOP_K,
                        threshold: float = RAG_RELEVANCE_THRESHOLD) -> dict:
        """回答戏曲相关问题。"""
        prepared = self.prepare_answer(
            question, top_k=top_k, threshold=threshold
        )
        references = prepared["references"]

        if qwen_model.is_loaded:
            logger.info(
                f"RAG 检索到 {len(references)} 条参考资料，"
                f"候选 {len(prepared['results'])} 条"
            )
            answer = qwen_model.chat(
                prepared["prompt"], prepared["system_prompt"]
            )
            source = "rag" if references else "general"
            note = (
                "此答案基于戏曲专业知识库生成，引用编号见 references"
                if references
                else "知识库未命中，此答案由大模型通用能力生成，仅供参考"
            )
        else:
            logger.info("千问模型未加载，返回检索结果或降级提示")
            answer = prepared["fallback_answer"]
            source = "rag" if references else "general"
            note = (
                "千问模型未启动，当前直接返回知识库检索片段"
                if references
                else "知识库未命中且千问模型未启动"
            )

        return {
            "answer": answer,
            "source": source,
            "references": references,
            "note": note,
            "retrieval": {
                "top_k": int(top_k),
                "threshold": float(threshold),
                "candidates": len(prepared["results"]),
                "used": len(references),
                "hybrid": True,
            },
        }

    def format_references_text(self, references: List[dict]) -> str:
        """生成前端展示用的来源列表。"""
        if not references:
            return ""
        lines = ["\n\n---\n参考来源："]
        for reference in references:
            citation_id = reference.get("citation_id", "")
            title = reference.get("title") or "未知"
            genre = reference.get("genre") or ""
            score = float(reference.get("score", 0.0))
            lines.append(
                f"[{citation_id}] {title}（{genre}） 相关度: {score:.4f}"
            )
        return "\n".join(lines)

    def analyze_video_with_rag(self, keyframe_paths: List[str],
                               visual_keywords: Optional[List[str]] = None) -> dict:
        """RAG 增强的视频结构化分析。"""
        self.ensure_knowledge_base()
        all_refs: List[dict] = []
        if visual_keywords:
            for keyword in visual_keywords:
                try:
                    all_refs.extend(knowledge_base.search(
                        keyword, top_k=max(3, RAG_TOP_K)
                    ))
                except Exception as exc:
                    logger.warning(f"视频分析关键词「{keyword}」检索失败: {exc}")

        merged_refs = self._merge_results(all_refs)[:10]
        knowledge_context, unique_refs = self._format_context(
            merged_refs, limit=10
        )

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
            "knowledge_refs": unique_refs,
        }


rag_service = RAGService()
