"""
CLIP 语义检索模块：
- 对关键帧图像进行编码并建立索引
- 对用户文本查询进行编码
- 计算文本-图像相似度，返回最匹配的关键帧
- 按视频隔离索引，避免多视频检索串索引
"""
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
from PIL import Image

from configs.settings import (
    CACHE_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED,
    CLIP_TOP_K, CLIP_SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)


class CLIPRetriever:
    """基于 CLIP 的语义图像检索器，支持按视频路由索引。"""

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        # 兼容原有单索引访问方式；当前激活的索引同时保存在 _indexes 中。
        self.index: Optional[faiss.IndexFlatIP] = None
        self.frame_metadata: List[dict] = []
        self._active_video_name: Optional[str] = None
        self._indexes: Dict[str, Tuple[faiss.IndexFlatIP, List[dict]]] = OrderedDict()
        self._lock = threading.RLock()
        self._device = "cpu"
        self._embedding_dim = None

    def _load_model(self):
        with self._lock:
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
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224).to(self._device)
                self._embedding_dim = self.model.encode_image(dummy).shape[-1]
            logger.info(f"CLIP 模型加载完成，嵌入维度: {self._embedding_dim}")

    def _encode_images_with_indices(
        self, image_paths: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, List[int]]:
        """编码图像，并返回成功编码图像在原始列表中的下标。"""
        import torch
        self._load_model()
        all_features = []
        encoded_indices: List[int] = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            batch_indices = []
            for offset, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    img = self.preprocess(img)
                    images.append(img)
                    batch_indices.append(i + offset)
                except Exception as e:
                    logger.warning(f"加载图像失败 {p}: {e}")
            if not images:
                continue
            image_tensor = torch.stack(images).to(self._device)
            with torch.no_grad():
                features = self.model.encode_image(image_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu().numpy())
                encoded_indices.extend(batch_indices)
        if not all_features:
            return np.array([]).reshape(0, self._embedding_dim or 512), []
        return np.vstack(all_features).astype(np.float32), encoded_indices

    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings, _ = self._encode_images_with_indices(image_paths, batch_size)
        return embeddings

    def encode_text(self, text: str) -> np.ndarray:
        import torch
        self._load_model()
        tokens = self.tokenizer([text]).to(self._device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32)

    def _set_active_index(
        self, video_name: str, index: faiss.IndexFlatIP, metadata: List[dict]
    ):
        self._indexes[video_name] = (index, metadata)
        self.index = index
        self.frame_metadata = metadata
        self._active_video_name = video_name

    def build_index(self, keyframes: List[dict],
                    video_name: Optional[str] = None) -> bool:
        """为指定视频构建索引；编码失败的帧不会进入元数据。"""
        if not keyframes:
            logger.warning("没有可建立索引的关键帧")
            return False
        image_paths = [kf["path"] for kf in keyframes]
        logger.info(f"为 {len(image_paths)} 个关键帧构建 CLIP 索引...")
        embeddings, encoded_indices = self._encode_images_with_indices(image_paths)
        if embeddings.shape[0] == 0:
            logger.warning("没有成功编码的图像，CLIP 索引未更新")
            return False
        if embeddings.shape[0] != len(encoded_indices):
            logger.error("CLIP 编码结果与有效图像下标数量不一致，拒绝更新索引")
            return False

        # 只保留编码成功的帧，避免 FAISS 向量与元数据错位。
        indexed_keyframes = [keyframes[i] for i in encoded_indices]
        new_index = faiss.IndexFlatIP(embeddings.shape[1])
        new_index.add(embeddings)
        skipped = len(keyframes) - len(indexed_keyframes)
        if skipped:
            logger.warning(f"跳过 {skipped} 个编码失败的关键帧")

        with self._lock:
            if video_name:
                self._set_active_index(video_name, new_index, indexed_keyframes)
            else:
                self.index = new_index
                self.frame_metadata = indexed_keyframes
                self._active_video_name = None
        logger.info(
            f"CLIP 索引构建完成，视频: {video_name or '(未命名)'}, "
            f"包含 {new_index.ntotal} 个向量"
        )
        return True

    @staticmethod
    def _index_paths(video_name: str) -> Tuple[Path, Path]:
        cache_dir = CACHE_DIR / "clip_indices"
        return (
            cache_dir / f"{video_name}.index",
            cache_dir / f"{video_name}_meta.json",
        )

    def save_index(self, video_name: str) -> bool:
        """保存指定视频的索引，不依赖当前激活的是哪个视频。"""
        with self._lock:
            entry = self._indexes.get(video_name)
            if entry is None and self._active_video_name == video_name and self.is_ready:
                entry = (self.index, self.frame_metadata)
            if entry is None:
                logger.warning(f"没有可保存的 CLIP 索引: {video_name}")
                return False
            index, metadata = entry
            if index is None or index.ntotal == 0:
                logger.warning(f"CLIP 索引为空，跳过保存: {video_name}")
                return False
            if index.ntotal != len(metadata):
                logger.error(
                    f"CLIP 索引与元数据数量不一致，拒绝保存: {video_name}, "
                    f"{index.ntotal} != {len(metadata)}"
                )
                return False

            index_path, meta_path = self._index_paths(video_name)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(index_path))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"CLIP 索引已保存: {video_name}, {index.ntotal} 个向量")
            return True

    def load_index(self, video_name: str) -> bool:
        """加载指定视频索引；校验失败时不会覆盖当前可用索引。"""
        with self._lock:
            cached = self._indexes.get(video_name)
            if cached is not None and cached[0] is not None and cached[0].ntotal > 0:
                self._set_active_index(video_name, cached[0], cached[1])
                return True

            index_path, meta_path = self._index_paths(video_name)
            if not index_path.exists() or not meta_path.exists():
                return False
            try:
                loaded_index = faiss.read_index(str(index_path))
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"加载 CLIP 索引失败 {video_name}: {e}")
                return False

            if loaded_index.ntotal == 0:
                logger.error(f"CLIP 索引为空，拒绝加载: {video_name}")
                return False
            if not isinstance(metadata, list) or loaded_index.ntotal != len(metadata):
                logger.error(
                    f"CLIP 索引与元数据数量不一致，拒绝加载: {video_name}, "
                    f"{loaded_index.ntotal} != {len(metadata) if isinstance(metadata, list) else 'invalid'}"
                )
                return False

            self._set_active_index(video_name, loaded_index, metadata)
            logger.info(f"加载 CLIP 索引: {video_name}, {loaded_index.ntotal} 个向量")
            return True

    def list_indexed_videos(self) -> List[str]:
        """列出内存中或磁盘上具有完整索引和元数据的视频。"""
        with self._lock:
            names = set(self._indexes.keys())
        cache_dir = CACHE_DIR / "clip_indices"
        if cache_dir.exists():
            for index_path in cache_dir.glob("*.index"):
                meta_path = cache_dir / f"{index_path.stem}_meta.json"
                if meta_path.exists():
                    names.add(index_path.stem)
        return sorted(names)

    @property
    def current_video_name(self) -> Optional[str]:
        return self._active_video_name

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def is_ready_for(self, video_name: str) -> bool:
        """判断指定视频的索引是否可用（不要求它是当前激活索引）。"""
        with self._lock:
            entry = self._indexes.get(video_name)
            return entry is not None and entry[0] is not None and entry[0].ntotal > 0

    def _search_single_index(
        self, query_features: np.ndarray, index: faiss.IndexFlatIP,
        metadata: List[dict], video_name: Optional[str],
        top_k: int, threshold: float
    ) -> List[dict]:
        scores, indices = index.search(
            query_features, min(top_k, index.ntotal)
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            meta = metadata[idx].copy()
            meta["score"] = float(score)
            if video_name is not None:
                meta["video_name"] = video_name
            results.append(meta)
        return results

    def search(self, query_text: str, top_k: int = CLIP_TOP_K,
               threshold: float = CLIP_SIMILARITY_THRESHOLD,
               video_name: Optional[str] = None) -> List[dict]:
        """
        检索关键帧。

        指定 video_name 时只搜索该视频；未指定时搜索全部已索引视频。
        """
        with self._lock:
            target_names: List[str]
            if video_name:
                if not self.load_index(video_name):
                    logger.warning(f"CLIP 索引不可用: {video_name}")
                    return []
                target_names = [video_name]
            else:
                target_names = self.list_indexed_videos()
                if not target_names:
                    if not self.is_ready:
                        logger.warning("尚无已建立的 CLIP 视频索引")
                        return []
                    # 兼容未指定视频名的旧调用方式。
                    query_features = self.encode_text(query_text)
                    results = self._search_single_index(
                        query_features, self.index, self.frame_metadata,
                        None, top_k, threshold
                    )
                    results.sort(key=lambda item: item["score"], reverse=True)
                    return results[:top_k]

            query_features = self.encode_text(query_text)
            results: List[dict] = []
            for name in target_names:
                if name not in self._indexes and not self.load_index(name):
                    continue
                entry = self._indexes.get(name)
                if entry is None:
                    continue
                index, metadata = entry
                results.extend(self._search_single_index(
                    query_features, index, metadata, name, top_k, threshold
                ))
            results.sort(key=lambda item: item["score"], reverse=True)
            return results[:top_k]


clip_retriever = CLIPRetriever()
