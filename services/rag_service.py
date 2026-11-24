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
_CITATION_BLOCK_RE = re.compile(
    r"\[(S\d+(?:\s*[,，、]\s*S\d+)*)\]",
    re.IGNORECASE,
)
_CITATION_TOKEN_RE = re.compile(r"S(\d+)", re.IGNORECASE)


class RAGService:
    """RAG 增强问答服务"""

    def ensure_knowledge_base(self):
        if not knowledge_base.ensure_ready(auto_build=False):
            logger.warning(
                "知识库索引不可用；请先运行 build_knowledge_base.py。"
                "问答将返回未命中提示或通用回答。"
            )

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
    def _reference_citation_numbers(references: List[dict]) -> set:
        """从 references 中提取允许出现的引用编号（只取数字部分）。"""
        allowed = set()
        for reference in references:
            match = _CITATION_TOKEN_RE.search(
                str(reference.get("citation_id", ""))
            )
            if match:
                allowed.add(int(match.group(1)))
        return allowed

    def _validate_answer_citations(
        self, answer: str, references: List[dict]
    ) -> Tuple[str, dict]:
        """校验答案中的 [S#] 编号，只保留 references 中真实存在的编号。

        提示词要求模型标注来源，但模型仍可能编造不存在的编号。这里做
        后端硬校验：无效编号直接从答案中删除，并返回结构化校验结果，
        避免仅靠提示词约束。
        """
        allowed_numbers = self._reference_citation_numbers(references)
        valid_cited: List[int] = []
        invalid_cited: List[int] = []

        def replace_block(match: re.Match) -> str:
            numbers = [
                int(token.group(1))
                for token in _CITATION_TOKEN_RE.finditer(match.group(1))
            ]
            kept: List[int] = []
            for number in numbers:
                if number in allowed_numbers:
                    if number not in valid_cited:
                        valid_cited.append(number)
                    if number not in kept:
                        kept.append(number)
                elif number not in invalid_cited:
                    invalid_cited.append(number)
            if not kept:
                return ""
            return "[" + "、".join(f"S{number}" for number in kept) + "]"

        cleaned = _CITATION_BLOCK_RE.sub(replace_block, answer or "")
        # 删除引用后清理多余空格，不改动正文其他内容。
        cleaned = re.sub(r"[ \t]+([，。；：！？,.!?;:])", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

        validation = {
            "valid_citation_ids": [
                f"S{number}" for number in sorted(allowed_numbers)
            ],
            "cited_citation_ids": [
                f"S{number}" for number in sorted(valid_cited)
            ],
            "invalid_citation_ids": [
                f"S{number}" for number in sorted(invalid_cited)
            ],
            "has_valid_citation": bool(valid_cited),
            "has_invalid_citation": bool(invalid_cited),
            "has_any_citation": bool(valid_cited or invalid_cited),
        }
        return cleaned, validation

    @staticmethod
    def _citation_warning(validation: dict, has_references: bool) -> Optional[str]:
        warnings: List[str] = []
        invalid_ids = validation.get("invalid_citation_ids", [])
        if invalid_ids:
            warnings.append(
                "已移除 {} 个不存在的引用编号（{}）".format(
                    len(invalid_ids), "、".join(invalid_ids)
                )
            )
        if has_references and not validation.get("has_valid_citation"):
            warnings.append("答案未包含有效资料引用，请核验关键事实")
        if not has_references and validation.get("has_any_citation"):
            warnings.append(
                "知识库未命中，答案中的引用编号已移除；"
                "内容仅代表模型通用知识"
            )
        return "；".join(warnings) if warnings else None

    @staticmethod
    def _summarize_retrieval(results: List[dict],
                             references: List[dict]) -> dict:
        """根据实际送进上下文的结果计算真实检索模式。

        优先信任每条结果的 ``retrieval_mode``。``dense_available`` 只表示
        本轮 Dense 通道是否产生了候选，不能据此把纯 BM25 结果误判为
        混合检索；只有缺少逐条模式时才回退到分数。
        """
        selected = references if references else results
        dense_used = False
        bm25_used = False

        for item in selected:
            mode = str(item.get("retrieval_mode") or "").lower()
            if mode not in {"hybrid", "dense", "bm25"}:
                mode = str(item.get("matched_by") or "").lower()
            if mode not in {"hybrid", "dense", "bm25"}:
                try:
                    dense_score = float(item.get("dense_score") or 0.0)
                    bm25_score = float(item.get("bm25_score") or 0.0)
                except (TypeError, ValueError):
                    dense_score = 0.0
                    bm25_score = 0.0
                if dense_score > 0 and bm25_score > 0:
                    mode = "hybrid"
                elif dense_score > 0:
                    mode = "dense"
                elif bm25_score > 0:
                    mode = "bm25"
                else:
                    mode = "unknown"

            if mode == "hybrid":
                dense_used = True
                bm25_used = True
            elif mode == "dense":
                dense_used = True
            elif mode == "bm25":
                bm25_used = True

        if dense_used and bm25_used:
            mode = "hybrid"
        elif dense_used:
            mode = "dense"
        elif bm25_used:
            mode = "bm25"
        elif selected:
            mode = "unknown"
        else:
            mode = "none"

        return {
            "mode": mode,
            "retrieval_mode": mode,
            "hybrid": mode == "hybrid",
            "dense_available": dense_used,
            "bm25_available": bm25_used,
        }

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
        retrieval = self._summarize_retrieval(
            prepared["results"], references
        )

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

        # 不把“有没有引用编号”完全交给提示词和模型自觉：后端做硬校验。
        answer, citation_validation = self._validate_answer_citations(
            answer, references
        )
        citation_warning = self._citation_warning(
            citation_validation, bool(references)
        )
        if citation_warning:
            note = f"{note}；{citation_warning}"

        embedding_status = knowledge_base.embedding_status
        embedding_error = knowledge_base.embedding_error
        if (
            references
            and retrieval["mode"] == "bm25"
            and embedding_status in {"unavailable", "incompatible"}
        ):
            note = f"{note}；Dense 嵌入模型不可用（本次使用 BM25 词法检索）"

        return {
            "answer": answer,
            "source": source,
            "references": references,
            "note": note,
            "citation_validation": citation_validation,
            "retrieval": {
                "top_k": int(top_k),
                "threshold": float(threshold),
                "candidates": len(prepared["results"]),
                "used": len(references),
                **retrieval,
                "embedding_status": embedding_status,
                "embedding_error": embedding_error,
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
