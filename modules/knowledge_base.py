"""
戏曲知识库构建与管理模块：
- 从原始数据目录读取 txt 文件作为知识来源
- 文本分块、向量化、存入 FAISS 索引
- 提供语义检索接口
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import faiss

from configs.settings import (
    OPERA_DATA_DIR, OPERA_GENRES,
    VECTOR_DB_DIR, RAW_KNOWLEDGE_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
    RAG_TOP_K, RAG_RELEVANCE_THRESHOLD
)

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """戏曲知识库：构建、管理与检索"""

    def __init__(self):
        self.embedding_model = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[dict] = []
        self._index_path = VECTOR_DB_DIR / "faiss_index.bin"
        self._docs_path = VECTOR_DB_DIR / "documents.json"
        self._embedding_dim = None

    def _load_embedding_model(self):
        if self.embedding_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"加载嵌入模型: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"嵌入模型加载完成，维度: {self._embedding_dim}")

    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> List[str]:
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def collect_knowledge_from_opera_data(self) -> List[dict]:
        """从原始戏曲数据目录收集知识文本"""
        all_docs = []
        for genre_dir in OPERA_DATA_DIR.iterdir():
            if not genre_dir.is_dir():
                continue
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
        """构建知识库 FAISS 索引"""
        self._load_embedding_model()
        raw_docs = self.collect_knowledge_from_opera_data()
        if additional_texts:
            raw_docs.extend(additional_texts)
        self.documents = []
        for doc in raw_docs:
            chunks = self._chunk_text(doc["text"])
            for chunk in chunks:
                self.documents.append({
                    "text": chunk,
                    "source": doc.get("source", ""),
                    "genre": doc.get("genre", ""),
                    "title": doc.get("title", ""),
                })
        if not self.documents:
            logger.warning("没有找到任何知识文本，创建空索引")
            self.index = faiss.IndexFlatIP(self._embedding_dim)
            return
        logger.info(f"共 {len(self.documents)} 个文本块，开始向量化...")
        texts = [d["text"] for d in self.documents]
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=True,
            normalize_embeddings=True, batch_size=64
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        logger.info(f"FAISS 索引构建完成，包含 {self.index.ntotal} 个向量")
        self.save()

    def save(self):
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self._index_path))
        with open(self._docs_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        logger.info("知识库索引和文档已保存")

    def load(self) -> bool:
        if not self._index_path.exists() or not self._docs_path.exists():
            logger.warning("未找到已有索引文件")
            return False
        self._load_embedding_model()
        self.index = faiss.read_index(str(self._index_path))
        with open(self._docs_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        logger.info(f"加载知识库: {self.index.ntotal} 个向量, {len(self.documents)} 个文档")
        return True

    def search(self, query: str, top_k: int = RAG_TOP_K,
               threshold: float = RAG_RELEVANCE_THRESHOLD) -> List[dict]:
        """语义检索知识库"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("知识库索引为空，请先构建索引")
            return []
        self._load_embedding_model()
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            doc = self.documents[idx]
            results.append({
                "text": doc["text"],
                "score": float(score),
                "source": doc.get("source", ""),
                "genre": doc.get("genre", ""),
                "title": doc.get("title", ""),
            })
        return results

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0


knowledge_base = KnowledgeBase()
