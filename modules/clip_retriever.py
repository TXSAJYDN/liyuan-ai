"""
CLIP 语义检索模块：
- 对关键帧图像进行编码并建立索引
- 对用户文本查询进行编码
- 计算文本-图像相似度，返回最匹配的关键帧
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import faiss
from PIL import Image

from configs.settings import (
    CACHE_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED,
    CLIP_TOP_K, CLIP_SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)


class CLIPRetriever:
    """基于 CLIP 的语义图像检索器"""

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.frame_metadata: List[dict] = []
        self._device = "cpu"
        self._embedding_dim = None

    def _load_model(self):
        if self.model is not None:
            return
        import torch
        import open_clip
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"加载 CLIP 模型: {CLIP_MODEL_NAME}, 设备: {self._device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=self._device
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self.model.eval()
        import torch
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self._device)
            self._embedding_dim = self.model.encode_image(dummy).shape[-1]
        logger.info(f"CLIP 模型加载完成，嵌入维度: {self._embedding_dim}")

    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        import torch
        self._load_model()
        all_features = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    img = self.preprocess(img)
                    images.append(img)
                except Exception as e:
                    logger.warning(f"加载图像失败 {p}: {e}")
                    continue
            if not images:
                continue
            image_tensor = torch.stack(images).to(self._device)
            with torch.no_grad():
                features = self.model.encode_image(image_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu().numpy())
        if not all_features:
            return np.array([]).reshape(0, self._embedding_dim or 512)
        return np.vstack(all_features).astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        import torch
        self._load_model()
        tokens = self.tokenizer([text]).to(self._device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32)

    def build_index(self, keyframes: List[dict]):
        self._load_model()
        image_paths = [kf["path"] for kf in keyframes]
        logger.info(f"为 {len(image_paths)} 个关键帧构建 CLIP 索引...")
        embeddings = self.encode_images(image_paths)
        if embeddings.shape[0] == 0:
            logger.warning("没有成功编码的图像")
            return
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.frame_metadata = keyframes
        logger.info(f"CLIP 索引构建完成，包含 {self.index.ntotal} 个向量")

    def search(self, query_text: str, top_k: int = CLIP_TOP_K,
               threshold: float = CLIP_SIMILARITY_THRESHOLD) -> List[dict]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("CLIP 索引为空")
            return []
        text_features = self.encode_text(query_text)
        scores, indices = self.index.search(text_features, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            meta = self.frame_metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results

    def save_index(self, video_name: str):
        cache_dir = CACHE_DIR / "clip_indices"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(cache_dir / f"{video_name}.index"))
        with open(cache_dir / f"{video_name}_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.frame_metadata, f, ensure_ascii=False, indent=2)

    def load_index(self, video_name: str) -> bool:
        cache_dir = CACHE_DIR / "clip_indices"
        index_path = cache_dir / f"{video_name}.index"
        meta_path = cache_dir / f"{video_name}_meta.json"
        if not index_path.exists() or not meta_path.exists():
            return False
        self._load_model()
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.frame_metadata = json.load(f)
        logger.info(f"加载 CLIP 索引: {video_name}, {self.index.ntotal} 个向量")
        return True

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0


clip_retriever = CLIPRetriever()
