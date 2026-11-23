"""
戏曲知识库构建与管理模块：
- 从原始数据目录读取 txt 文件作为知识来源
- 按自然边界递归分块、向量化、存入 FAISS 索引
- 同时维护 BM25 词法索引，提供 dense + BM25 混合检索
- 校验索引文件与文档/嵌入模型是否一致，避免加载脏索引
"""
import json
import logging
import math
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from configs.settings import (
    OPERA_DATA_DIR, OPERA_GENRES,
    VECTOR_DB_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
    RAG_TOP_K, RAG_RELEVANCE_THRESHOLD,
    RAG_DENSE_CANDIDATE_K, RAG_BM25_CANDIDATE_K,
    RAG_HYBRID_DENSE_WEIGHT, RAG_HYBRID_BM25_WEIGHT,
    RAG_DEDUP_THRESHOLD,
)

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*|[\u4e00-\u9fff]+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

# 中文戏曲术语到英文/拼音的轻量扩展。BM25 只使用这些词做词法召回，
# dense 向量仍然使用用户原始问题，避免扩展词改变语义方向。
_LEXICAL_EXPANSIONS = {
    "西皮": ("xipi",),
    "二黄": ("erhuang", "er huang"),
    "二簧": ("erhuang", "er huang"),
    "起霸": ("qiba", "qi ba"),
    "亮相": ("liangxiang", "pose", "posing"),
    "云手": ("yunshou", "cloud hands"),
    "水袖": ("water sleeves", "shuixiu"),
    "台步": ("stage walk", "taibu"),
    "兰花指": ("orchid fingers", "lanhua finger"),
    "髯口": ("beard", "rankou"),
    "翎子": ("pheasant feathers", "lingzi"),
    "脸谱": ("facial makeup", "facial mask", "lianpu"),
    "蟒袍": ("dragon robe", "mangpao"),
    "厚底靴": ("thick-soled boots", "hou di xue"),
    "行当": ("role type", "hangdang"),
    "生旦净丑": ("sheng dan jing chou", "role types"),
    "生": ("sheng", "male role"),
    "旦": ("dan", "female role"),
    "净": ("jing", "painted face role"),
    "丑": ("chou", "clown role"),
    "唱腔": ("aria", "singing style", "changqiang"),
    "板式": ("rhythm", "metre", "ban shi"),
    "念白": ("spoken dialogue", "nianbai"),
    "身段": ("body movement", "shen duan"),
    "程式": ("stylized movement", "convention"),
    "武打": ("acrobatic fighting", "martial arts"),
    "把子": ("weapons choreography", "bazi"),
    "京剧": ("peking opera", "jingju", "beijing opera"),
    "昆曲": ("kunqu", "kun opera"),
    "越剧": ("yue opera", "yueju"),
    "梆子": ("bangzi", "clapper opera"),
    "高腔": ("gaoqiang", "high pitch"),
    "花灯戏": ("huadeng opera", "lantern opera"),
    "二人台": ("errentai", "duet stage"),
    "样板戏": ("model opera", "yangbanxi"),
    "折子戏": ("opera excerpt", "zhezixi"),
    "剧目": ("repertoire", "play title"),
    "服饰": ("costume", "clothing"),
    "道具": ("props", "stage property"),
    "锣鼓": ("gongs and drums", "luogu"),
    "胡琴": ("huqin", "fiddle"),
    "梅兰芳": ("mei lanfang",),
    "程砚秋": ("cheng yanqiu",),
    "荀慧生": ("xun huisheng",),
    "尚小云": ("shang xiaoyun",),
}

_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "请", "请问", "什么", "是", "的", "了", "和", "与", "及",
    "有", "哪些", "如何", "怎么", "为什么", "介绍", "一下", "区别",
}


class KnowledgeBase:
    """戏曲知识库：构建、管理与混合检索"""

    _SEPARATORS = (
        "\n\n", "\n",
        "。", "！", "？", "；",
        ". ", "! ", "? ", "; ",
        "，", ", ",
        " ",
    )
    _BM25_K1 = 1.5
    _BM25_B = 0.75

    def __init__(self):
        self.embedding_model = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[dict] = []
        self._index_path = VECTOR_DB_DIR / "faiss_index.bin"
        self._docs_path = VECTOR_DB_DIR / "documents.json"
        self._manifest_path = VECTOR_DB_DIR / "index_manifest.json"
        self._embedding_dim: Optional[int] = None
        self._lock = threading.RLock()

        # BM25 运行时状态，加载/构建索引时由 documents 重建。
        self._bm25_tf: List[Counter] = []
        self._bm25_lengths: List[int] = []
        self._bm25_idf: Dict[str, float] = {}
        self._bm25_avgdl = 0.0

    def _load_embedding_model(self):
        if self.embedding_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"加载嵌入模型: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"嵌入模型加载完成，维度: {self._embedding_dim}")

    # ------------------------------------------------------------------
    # 分块
    # ------------------------------------------------------------------
    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> List[str]:
        """按段落、换行、句子、逗号、空格的优先级递归分块。"""
        text = (text or "").strip()
        if not text:
            return []

        chunk_size = max(1, int(chunk_size))
        overlap = max(0, min(int(overlap), chunk_size - 1))
        if len(text) <= chunk_size:
            return [text]

        chunks = self._split_recursive(text, self._SEPARATORS, chunk_size, overlap)
        # 去掉空白块并保持顺序；极端情况下（没有自然边界）允许字符兜底。
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    def _split_recursive(self, text: str, separators: Sequence[str],
                         chunk_size: int, overlap: int) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        for index, separator in enumerate(separators):
            if separator not in text:
                continue
            parts = text.split(separator)
            pieces: List[str] = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if len(part) > chunk_size:
                    pieces.extend(self._split_recursive(
                        part, separators[index + 1:], chunk_size, overlap
                    ))
                else:
                    pieces.append(part)

            if len(pieces) <= 1:
                continue
            return self._merge_parts(pieces, separator, chunk_size, overlap)

        # 字符兜底：只有文本完全不存在自然边界时才会走到这里。
        return [
            text[start:start + chunk_size].strip()
            for start in range(0, len(text), chunk_size)
            if text[start:start + chunk_size].strip()
        ]

    @staticmethod
    def _joined_length(parts: Sequence[str], separator: str) -> int:
        if not parts:
            return 0
        return sum(len(part) for part in parts) + len(separator) * (len(parts) - 1)

    def _merge_parts(self, parts: Sequence[str], separator: str,
                     chunk_size: int, overlap: int) -> List[str]:
        chunks: List[str] = []
        current: List[str] = []
        separator_len = len(separator)

        for part in parts:
            projected = self._joined_length(current, separator) + separator_len + len(part) if current else len(part)
            if current and projected > chunk_size:
                doc = separator.join(current).strip()
                if doc:
                    chunks.append(doc)

                # 保留不超过 overlap 的尾部作为下一块的上下文，并确保新 part 能放下。
                while current:
                    current_len = self._joined_length(current, separator)
                    fits_next = current_len + separator_len + len(part) <= chunk_size
                    if current_len <= overlap and fits_next:
                        break
                    current.pop(0)

            current.append(part)

        if current:
            doc = separator.join(current).strip()
            if doc:
                chunks.append(doc)
        return chunks

    # ------------------------------------------------------------------
    # 词法索引（轻量 BM25）
    # ------------------------------------------------------------------
    def _tokenize(self, text: str, for_query: bool = False) -> List[str]:
        tokens: List[str] = []
        lowered = (text or "").lower()
        for match in _TOKEN_RE.finditer(lowered):
            token = match.group(0)
            if _CJK_RE.match(token):
                if len(token) == 1:
                    if not for_query or token not in _QUERY_STOPWORDS:
                        tokens.append(token)
                else:
                    # 中文按 2-gram 切分；文档额外保留单字，支持单字查询。
                    if not for_query:
                        tokens.extend(token)
                    tokens.extend(token[i:i + 2] for i in range(len(token) - 1))
            else:
                if for_query and token in _QUERY_STOPWORDS:
                    continue
                tokens.append(token)
        return tokens

    def _expand_query_for_lexical(self, query: str) -> str:
        expansions: List[str] = []
        for term, aliases in _LEXICAL_EXPANSIONS.items():
            if term in query:
                expansions.extend(aliases)
        if not expansions:
            return query
        # 去重并保持顺序。
        return query + " " + " ".join(dict.fromkeys(expansions))

    def _build_bm25_index(self, documents: Sequence[dict]):
        tokenized_docs = [
            self._tokenize(doc.get("text", ""), for_query=False)
            for doc in documents
        ]
        self._bm25_tf = [Counter(tokens) for tokens in tokenized_docs]
        self._bm25_lengths = [len(tokens) for tokens in tokenized_docs]
        self._bm25_avgdl = (
            sum(self._bm25_lengths) / len(self._bm25_lengths)
            if self._bm25_lengths else 0.0
        )

        document_frequency: Counter = Counter()
        for term_freq in self._bm25_tf:
            document_frequency.update(term_freq.keys())

        total_docs = len(documents)
        self._bm25_idf = {
            term: math.log(1.0 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def _bm25_scores(self, query: str) -> Dict[int, float]:
        if not self.documents or not self._bm25_tf or self._bm25_avgdl <= 0:
            return {}

        query_terms = list(dict.fromkeys(
            self._tokenize(self._expand_query_for_lexical(query), for_query=True)
        ))
        if not query_terms:
            return {}

        scores: Dict[int, float] = {}
        for index, term_freq in enumerate(self._bm25_tf):
            doc_len = self._bm25_lengths[index]
            score = 0.0
            for term in query_terms:
                freq = term_freq.get(term, 0)
                if not freq:
                    continue
                idf = self._bm25_idf.get(term, 0.0)
                denominator = freq + self._BM25_K1 * (
                    1.0 - self._BM25_B + self._BM25_B * doc_len / self._bm25_avgdl
                )
                score += idf * freq * (self._BM25_K1 + 1.0) / denominator
            if score > 0:
                scores[index] = float(score)
        return scores

    # ------------------------------------------------------------------
    # 去重
    # ------------------------------------------------------------------
    @staticmethod
    def _normalized_text(text: str) -> str:
        return _WHITESPACE_RE.sub(" ", (text or "").lower()).strip()

    def _is_near_duplicate(self, left: str, right: str,
                           threshold: float = RAG_DEDUP_THRESHOLD) -> bool:
        left_norm = self._normalized_text(left)
        right_norm = self._normalized_text(right)
        if not left_norm or not right_norm:
            return left_norm == right_norm
        if left_norm == right_norm:
            return True

        shorter, longer = sorted((left_norm, right_norm), key=len)
        if len(shorter) >= 40 and shorter in longer and len(shorter) / len(longer) >= 0.75:
            return True

        left_tokens = set(self._tokenize(left_norm))
        right_tokens = set(self._tokenize(right_norm))
        if not left_tokens or not right_tokens:
            return False
        union = left_tokens | right_tokens
        jaccard = len(left_tokens & right_tokens) / len(union)
        return jaccard >= threshold

    def _dedupe_ranked(self, ranked: Sequence[dict], limit: int) -> List[dict]:
        unique: List[dict] = []
        for item in ranked:
            if any(
                self._is_near_duplicate(item.get("text", ""), kept.get("text", ""))
                for kept in unique
            ):
                continue
            unique.append(item)
            if len(unique) >= limit:
                break
        return unique

    # ------------------------------------------------------------------
    # 建库 / 保存 / 加载
    # ------------------------------------------------------------------
    def collect_knowledge_from_opera_data(self) -> List[dict]:
        """从原始戏曲数据目录收集知识文本（只读访问）。"""
        all_docs: List[dict] = []
        if not OPERA_DATA_DIR.exists():
            logger.warning(f"戏曲数据目录不存在: {OPERA_DATA_DIR}")
            return all_docs

        try:
            genre_dirs = sorted(
                (path for path in OPERA_DATA_DIR.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            )
        except OSError as exc:
            logger.warning(f"读取戏曲数据目录失败: {exc}")
            return all_docs

        for genre_dir in genre_dirs:
            genre_key = genre_dir.name
            genre_name = OPERA_GENRES.get(genre_key, genre_key)
            for txt_file in sorted(genre_dir.glob("*.txt")):
                try:
                    content = txt_file.read_text(encoding="utf-8").strip()
                    if not content:
                        continue
                    all_docs.append({
                        "text": content,
                        "source": str(txt_file),
                        "genre": genre_name,
                        "title": txt_file.stem,
                    })
                except Exception as e:
                    logger.warning(f"读取 {txt_file} 失败: {e}")
        logger.info(f"从戏曲数据目录收集了 {len(all_docs)} 个知识文档")
        return all_docs

    def build_index(self, additional_texts: Optional[List[dict]] = None):
        """构建知识库 FAISS + BM25 索引。"""
        with self._lock:
            self._load_embedding_model()
            raw_docs = self.collect_knowledge_from_opera_data()
            if additional_texts:
                raw_docs.extend(additional_texts)

            new_documents: List[dict] = []
            for doc in raw_docs:
                text = (doc.get("text") or "").strip()
                if not text:
                    continue
                chunks = self._chunk_text(text)
                for chunk_index, chunk in enumerate(chunks):
                    new_documents.append({
                        "text": chunk,
                        "source": doc.get("source", ""),
                        "genre": doc.get("genre", ""),
                        "title": doc.get("title", ""),
                        "chunk_index": chunk_index,
                    })

            if not new_documents:
                # 数据源不可用时绝不能覆盖已有索引；否则一次离线启动就可能把
                # 可用的知识库清空成 0 条向量。
                logger.warning(
                    "没有找到任何知识文本，保留现有内存/磁盘索引不变"
                )
                return

            logger.info(f"共 {len(new_documents)} 个文本块，开始向量化...")
            embeddings = self.embedding_model.encode(
                [doc["text"] for doc in new_documents],
                show_progress_bar=True,
                normalize_embeddings=True,
                batch_size=64,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim != 2:
                raise ValueError(f"嵌入向量维度异常: {embeddings.shape}")
            if embeddings.shape[0] != len(new_documents):
                raise ValueError(
                    f"嵌入向量数 {embeddings.shape[0]} 与文档块数 {len(new_documents)} 不一致"
                )

            new_index = faiss.IndexFlatIP(embeddings.shape[1])
            new_index.add(embeddings)

            # 全部校验通过后再替换当前可用索引，避免构建失败破坏在线检索。
            self.documents = new_documents
            self.index = new_index
            self._embedding_dim = int(embeddings.shape[1])
            self._build_bm25_index(new_documents)
            logger.info(f"FAISS 索引构建完成，包含 {self.index.ntotal} 个向量")
            self.save()

    def save(self):
        with self._lock:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            self._docs_path.parent.mkdir(parents=True, exist_ok=True)
            index_tmp = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
            if self.index is not None:
                faiss.write_index(self.index, str(index_tmp))

            docs_tmp = self._docs_path.with_suffix(self._docs_path.suffix + ".tmp")
            with open(docs_tmp, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            docs_tmp.replace(self._docs_path)

            manifest = {
                "format_version": 2,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_dim": int(self._embedding_dim or (self.index.d if self.index else 0)),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "document_count": len(self.documents),
                "vector_count": int(self.index.ntotal) if self.index is not None else 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_tmp = self._manifest_path.with_suffix(
                self._manifest_path.suffix + ".tmp"
            )
            with open(manifest_tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            # 三份文件都先写入临时文件，最后再替换，减少写到一半中断导致
            # 索引、文档和 manifest 互相不一致的概率。
            if self.index is not None:
                index_tmp.replace(self._index_path)
            docs_tmp.replace(self._docs_path)
            manifest_tmp.replace(self._manifest_path)
            logger.info("知识库索引、文档和 manifest 已保存")

    def load(self) -> bool:
        """加载并校验索引；向量数、文档数或模型维度不一致时拒绝使用。"""
        with self._lock:
            if not self._index_path.exists() or not self._docs_path.exists():
                logger.warning("未找到已有索引文件")
                return False

            try:
                with open(self._docs_path, "r", encoding="utf-8") as f:
                    documents = json.load(f)
                if not isinstance(documents, list):
                    raise ValueError("documents.json 不是列表")

                index = faiss.read_index(str(self._index_path))
                if index.ntotal != len(documents):
                    raise ValueError(
                        f"索引向量数 {index.ntotal} 与文档数 {len(documents)} 不一致"
                    )

                self._load_embedding_model()
                model_dim = self.embedding_model.get_sentence_embedding_dimension()
                if int(index.d) != int(model_dim):
                    raise ValueError(
                        f"索引维度 {index.d} 与嵌入模型维度 {model_dim} 不一致"
                    )

                if self._manifest_path.exists():
                    with open(self._manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    indexed_model = manifest.get("embedding_model")
                    if indexed_model and indexed_model != EMBEDDING_MODEL_NAME:
                        raise ValueError(
                            f"索引由模型 {indexed_model} 构建，与当前模型 "
                            f"{EMBEDDING_MODEL_NAME} 不一致"
                        )
                    if int(manifest.get("embedding_dim", index.d)) != int(index.d):
                        raise ValueError("manifest 中的嵌入维度与索引不一致")
                    if int(manifest.get("document_count", len(documents))) != len(documents):
                        raise ValueError("manifest 中的文档数与 documents.json 不一致")
                    if int(manifest.get("vector_count", index.ntotal)) != index.ntotal:
                        raise ValueError("manifest 中的向量数与索引不一致")
                    manifest_chunk_size = manifest.get("chunk_size")
                    manifest_chunk_overlap = manifest.get("chunk_overlap")
                    if (
                        (manifest_chunk_size is not None
                         and int(manifest_chunk_size) != CHUNK_SIZE)
                        or (manifest_chunk_overlap is not None
                            and int(manifest_chunk_overlap) != CHUNK_OVERLAP)
                    ):
                        logger.warning(
                            "索引分块参数 (chunk_size=%s, chunk_overlap=%s) 与当前配置 "
                            "(%s, %s) 不一致，建议重建索引",
                            manifest_chunk_size, manifest_chunk_overlap,
                            CHUNK_SIZE, CHUNK_OVERLAP,
                        )
                else:
                    logger.warning(
                        "索引缺少 manifest，已兼容加载；建议重建索引以应用新的分块配置"
                    )

                self.index = index
                self.documents = documents
                self._embedding_dim = int(model_dim)
                self._build_bm25_index(documents)
                logger.info(
                    f"加载知识库: {self.index.ntotal} 个向量, "
                    f"{len(self.documents)} 个文档"
                )
                return True
            except Exception as e:
                logger.error(f"知识库索引加载失败，拒绝使用脏索引: {e}")
                self.index = None
                self.documents = []
                self._build_bm25_index([])
                return False

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = RAG_TOP_K,
               threshold: float = RAG_RELEVANCE_THRESHOLD) -> List[dict]:
        """dense + BM25 混合检索。

        threshold 只约束向量候选；BM25 精确词命中的候选不会被低跨语言
        dense 分数误过滤。最终 score 为两路归一化分数的加权融合。
        """
        query = (query or "").strip()
        top_k = int(top_k)
        if not query or top_k <= 0:
            return []

        with self._lock:
            if not self.is_ready:
                logger.warning("知识库索引为空或文档/向量未对齐，请先构建索引")
                return []

            self._load_embedding_model()
            query_embedding = np.asarray(
                self.embedding_model.encode(
                    [query], normalize_embeddings=True
                ),
                dtype=np.float32,
            )
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            dense_k = min(max(top_k, int(RAG_DENSE_CANDIDATE_K)), self.index.ntotal)
            scores, indices = self.index.search(query_embedding, dense_k)
            dense_scores: Dict[int, float] = {}
            for score, index in zip(scores[0], indices[0]):
                if index >= 0:
                    dense_scores[int(index)] = float(score)

            bm25_scores = self._bm25_scores(query)
            bm25_ranked = sorted(
                bm25_scores.items(), key=lambda item: item[1], reverse=True
            )[:max(top_k, int(RAG_BM25_CANDIDATE_K))]
            if bm25_ranked:
                bm25_max = bm25_ranked[0][1]
                bm25_ranked = [
                    (index, score) for index, score in bm25_ranked
                    if score >= bm25_max * 0.1
                ]

            candidate_indices = {
                index for index, score in dense_scores.items()
                if score >= float(threshold)
            }
            candidate_indices.update(index for index, _ in bm25_ranked)
            if not candidate_indices:
                return []

            bm25_max = max(
                (bm25_scores.get(index, 0.0) for index in candidate_indices),
                default=0.0,
            )
            ranked: List[dict] = []
            for index in candidate_indices:
                dense_score = float(dense_scores.get(index, 0.0))
                bm25_score = float(bm25_scores.get(index, 0.0))
                dense_normalized = max(0.0, min(1.0, dense_score))
                bm25_normalized = bm25_score / bm25_max if bm25_max > 0 else 0.0
                final_score = (
                    RAG_HYBRID_DENSE_WEIGHT * dense_normalized
                    + RAG_HYBRID_BM25_WEIGHT * bm25_normalized
                )
                if dense_score >= float(threshold) and bm25_score > 0:
                    matched_by = "hybrid"
                elif bm25_score > 0:
                    matched_by = "bm25"
                else:
                    matched_by = "dense"

                doc = self.documents[index]
                ranked.append({
                    "text": doc.get("text", ""),
                    "score": float(final_score),
                    "dense_score": dense_score,
                    "bm25_score": bm25_score,
                    "matched_by": matched_by,
                    "source": doc.get("source", ""),
                    "genre": doc.get("genre", ""),
                    "title": doc.get("title", ""),
                    "chunk_index": doc.get("chunk_index"),
                })

            ranked.sort(
                key=lambda item: (
                    item["score"], item["dense_score"], item["bm25_score"]
                ),
                reverse=True,
            )
            # 多取一些候选再去重，避免近似重复块挤占 top_k。
            candidates = ranked[:max(top_k * 3, top_k + 5)]
            return self._dedupe_ranked(candidates, top_k)[:top_k]

    @property
    def is_ready(self) -> bool:
        return (
            self.index is not None
            and self.index.ntotal > 0
            and len(self.documents) == self.index.ntotal
        )


knowledge_base = KnowledgeBase()
