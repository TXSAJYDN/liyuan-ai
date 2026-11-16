"""CLIP 多视频索引路由回归测试。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

import faiss
import numpy as np

import modules.clip_retriever as clip_module
from modules.clip_retriever import CLIPRetriever


class ClipRetrieverRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_patch = patch.object(
            clip_module, "CACHE_DIR", Path(self.temp_dir.name)
        )
        self.cache_patch.start()
        self.retriever = CLIPRetriever()

    def tearDown(self):
        self.cache_patch.stop()
        self.temp_dir.cleanup()

    def _build_and_save(self, retriever, video_name, vectors,
                        valid_indices=None):
        keyframes = [
            {"path": f"/frames/{video_name}/frame_{i}.jpg", "timestamp": i}
            for i in range(len(vectors))
        ]
        embeddings = np.asarray(vectors, dtype=np.float32)
        if valid_indices is None:
            valid_indices = list(range(len(keyframes)))
        with patch.object(
            retriever, "_encode_images_with_indices",
            return_value=(embeddings, valid_indices),
        ):
            self.assertTrue(
                retriever.build_index(keyframes, video_name=video_name)
            )
        self.assertTrue(retriever.save_index(video_name))
        return keyframes

    def test_search_routes_to_requested_video(self):
        self._build_and_save(self.retriever, "video_a", [[1.0, 0.0]])
        self._build_and_save(self.retriever, "video_b", [[0.0, 1.0]])

        with patch.object(
            self.retriever, "encode_text",
            return_value=np.array([[1.0, 0.0]], dtype=np.float32),
        ):
            result_a = self.retriever.search(
                "test", top_k=1, threshold=0.0, video_name="video_a"
            )
            result_b = self.retriever.search(
                "test", top_k=1, threshold=0.0, video_name="video_b"
            )
            result_all = self.retriever.search(
                "test", top_k=2, threshold=0.0
            )

        self.assertEqual(result_a[0]["video_name"], "video_a")
        self.assertIn("/video_a/", result_a[0]["path"])
        self.assertEqual(result_b[0]["video_name"], "video_b")
        self.assertIn("/video_b/", result_b[0]["path"])
        self.assertEqual(
            {item["video_name"] for item in result_all},
            {"video_a", "video_b"},
        )

    def test_build_index_keeps_metadata_aligned(self):
        keyframes = [
            {"path": "/frames/c/0.jpg", "timestamp": 0},
            {"path": "/frames/c/1.jpg", "timestamp": 1},
            {"path": "/frames/c/2.jpg", "timestamp": 2},
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with patch.object(
            self.retriever, "_encode_images_with_indices",
            return_value=(embeddings, [0, 2]),
        ):
            self.assertTrue(
                self.retriever.build_index(keyframes, video_name="video_c")
            )

        self.assertEqual(self.retriever.index.ntotal, 2)
        self.assertEqual(
            [item["path"] for item in self.retriever.frame_metadata],
            ["/frames/c/0.jpg", "/frames/c/2.jpg"],
        )

    def test_invalid_index_does_not_replace_current_index(self):
        self._build_and_save(self.retriever, "video_a", [[1.0, 0.0]])
        self.assertTrue(self.retriever.load_index("video_a"))

        index_dir = Path(self.temp_dir.name) / "clip_indices"
        faiss.write_index(
            faiss.IndexFlatIP(2), str(index_dir / "broken.index")
        )
        (index_dir / "broken_meta.json").write_text("[]", encoding="utf-8")

        self.assertFalse(self.retriever.load_index("broken"))
        self.assertEqual(self.retriever.current_video_name, "video_a")
        self.assertTrue(self.retriever.is_ready_for("video_a"))


class PipelineClipRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from modules.qwen_model import QwenModel

        cls.qwen_patch = patch.object(QwenModel, "load_model", return_value=None)
        cls.qwen_patch.start()
        import services.pipeline as pipeline_module
        cls.pipeline_module = pipeline_module

    @classmethod
    def tearDownClass(cls):
        cls.qwen_patch.stop()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_patch = patch.object(
            clip_module, "CACHE_DIR", Path(self.temp_dir.name)
        )
        self.cache_patch.start()
        self.retriever = CLIPRetriever()
        self.pipeline = object.__new__(self.pipeline_module.Pipeline)

    def tearDown(self):
        self.cache_patch.stop()
        self.temp_dir.cleanup()

    def test_cached_video_does_not_reuse_another_video_index(self):
        retriever = self.retriever
        keyframes_a = [
            {"path": "/frames/video_a/frame_0.jpg", "timestamp": 0}
        ]
        with patch.object(
            retriever, "_encode_images_with_indices",
            return_value=(
                np.array([[1.0, 0.0]], dtype=np.float32), [0]
            ),
        ):
            self.assertTrue(
                retriever.build_index(keyframes_a, video_name="video_a")
            )
        self.assertTrue(retriever.save_index("video_a"))
        self.assertTrue(retriever.load_index("video_a"))

        keyframes_b = [
            {"path": "/frames/video_b/frame_0.jpg", "timestamp": 0}
        ]
        cached_result = {
            "video_path": "video_b.mp4",
            "video_info": {},
            "segments": [],
            "keyframes": keyframes_b,
        }
        with patch.object(
            self.pipeline_module, "clip_retriever", retriever
        ), patch.object(
            self.pipeline_module.video_processor,
            "load_cached_result", return_value=cached_result,
        ), patch.object(
            retriever, "_encode_images_with_indices",
            return_value=(
                np.array([[0.0, 1.0]], dtype=np.float32), [0]
            ),
        ):
            result = self.pipeline.process_uploaded_video("video_b.mp4")

        self.assertTrue(result["clip_index_ready"])
        self.assertEqual(retriever.current_video_name, "video_b")
        self.assertTrue((Path(self.temp_dir.name) / "clip_indices" / "video_b.index").exists())

    def test_semantic_search_loads_requested_video_index(self):
        self._write_index("video_a", [1.0, 0.0])
        self._write_index("video_b", [0.0, 1.0])
        self.assertTrue(self.retriever.load_index("video_a"))

        with patch.object(
            self.pipeline_module, "clip_retriever", self.retriever
        ), patch.object(
            self.retriever, "encode_text",
            return_value=np.array([[0.0, 1.0]], dtype=np.float32),
        ), patch.object(
            type(self.pipeline_module.qwen_model),
            "is_loaded",
            new_callable=PropertyMock,
            return_value=False,
        ):
            results = self.pipeline.semantic_search("test", video_name="video_b")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["video_name"], "video_b")
        self.assertIn("/video_b/", results[0]["path"])

    def _write_index(self, video_name, vector):
        keyframes = [
            {"path": f"/frames/{video_name}/frame_0.jpg", "timestamp": 0}
        ]
        with patch.object(
            self.retriever, "_encode_images_with_indices",
            return_value=(np.array([vector], dtype=np.float32), [0]),
        ):
            self.retriever.build_index(keyframes, video_name=video_name)
        self.retriever.save_index(video_name)


if __name__ == "__main__":
    unittest.main()
