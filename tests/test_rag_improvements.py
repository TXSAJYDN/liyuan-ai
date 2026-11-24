"""RAG 混合检索、分块、引用和参数贯通回归测试。"""
import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

import faiss
import numpy as np

import services.rag_service as rag_module
from modules.knowledge_base import KnowledgeBase
from services.rag_service import RAGService


class FakeEmbeddingModel:
    """按输入文本返回预设向量的假嵌入模型，避免测试下载模型。"""

    def __init__(self, dimension=4, vectors=None, default=None):
        self.dimension = dimension
        self.vectors = vectors or {}
        self.default = default if default is not None else [1.0, 0.0, 0.0, 0.0]

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, normalize_embeddings=True,
               show_progress_bar=False, batch_size=64):
        vectors = [self.vectors.get(text, self.default) for text in texts]
        return np.asarray(vectors, dtype=np.float32)


def make_ready_kb(documents, vectors, dimension=4):
    kb = KnowledgeBase()
    kb.embedding_model = FakeEmbeddingModel(dimension)
    kb._embedding_dim = dimension
    kb.documents = documents
    kb.index = faiss.IndexFlatIP(dimension)
    kb.index.add(np.asarray(vectors, dtype=np.float32))
    kb._build_bm25_index(documents)
    return kb


class ChunkingTests(unittest.TestCase):
    def test_chunking_does_not_cut_english_words(self):
        kb = KnowledgeBase()
        words = [f"word{i}" for i in range(400)]
        text = " ".join(words)
        chunks = kb._chunk_text(text, chunk_size=120, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        original_words = set(words)
        for chunk in chunks:
            self.assertTrue(set(chunk.split()).issubset(original_words))

    def test_chunking_keeps_natural_boundaries_and_overlap(self):
        kb = KnowledgeBase()
        text = "第一段内容。第二段内容。第三段内容。第四段内容。"
        chunks = kb._chunk_text(text, chunk_size=12, overlap=6)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 12 for chunk in chunks))
        first_last_part = chunks[0].split("。")[-1]
        self.assertIn(first_last_part, chunks[1])

    def test_overlap_is_clamped_to_avoid_infinite_loop(self):
        kb = KnowledgeBase()
        chunks = kb._chunk_text("abc def ghi", chunk_size=5, overlap=99)
        self.assertEqual(chunks, ["abc", "def", "ghi"])


class HybridSearchTests(unittest.TestCase):
    def test_bm25_recalls_exact_term_even_when_dense_score_is_low(self):
        documents = [
            {
                "text": "Qiba is a stylized martial posture in Chinese opera.",
                "title": "Qiba",
                "genre": "京剧",
            },
            {
                "text": "Water sleeves express emotion on stage.",
                "title": "Water sleeves",
                "genre": "京剧",
            },
        ]
        kb = make_ready_kb(
            documents,
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        )
        query = "什么是起霸？"
        kb.embedding_model = FakeEmbeddingModel(
            vectors={query: [1.0, 0.0, 0.0, 0.0]}
        )

        results = kb.search(query, top_k=3, threshold=0.4)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Qiba")
        self.assertEqual(results[0]["matched_by"], "bm25")
        self.assertEqual(results[0]["retrieval_mode"], "bm25")
        self.assertEqual(results[0]["dense_score"], 0.0)
        self.assertGreater(results[0]["bm25_score"], 0.0)
        # 纯 BM25 降级不再乘 0.35 融合权重，唯一候选的归一化分数应为 1。
        self.assertAlmostEqual(results[0]["score"], 1.0)

    def test_dense_threshold_rejects_unrelated_results(self):
        documents = [{
            "text": "Completely unrelated content about weather.",
            "title": "Weather",
            "genre": "其他",
        }]
        kb = make_ready_kb(documents, [[0.0, 1.0, 0.0, 0.0]])
        query = "戏曲武功动作"
        kb.embedding_model = FakeEmbeddingModel(
            vectors={query: [1.0, 0.0, 0.0, 0.0]}
        )

        self.assertEqual(kb.search(query, top_k=3, threshold=0.4), [])

    def test_search_deduplicates_identical_chunks(self):
        text = "Qiba is a stylized martial posture used at the start of a performance."
        documents = [
            {"text": text, "title": "A", "genre": "京剧"},
            {"text": text, "title": "B", "genre": "京剧"},
            {"text": "Water sleeves express emotion.", "title": "C", "genre": "京剧"},
        ]
        kb = make_ready_kb(
            documents,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.99, 0.01, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
            ],
        )
        query = "qiba posture"
        kb.embedding_model = FakeEmbeddingModel(
            vectors={query: [1.0, 0.0, 0.0, 0.0]}
        )

        results = kb.search(query, top_k=2, threshold=0.4)

        self.assertEqual(len(results), 2)
        titles = [item["title"] for item in results]
        self.assertEqual(len(set(titles)), 2)
        self.assertNotIn("B", titles)


class IndexValidationTests(unittest.TestCase):
    def test_load_rejects_vector_document_count_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = root / "faiss_index.bin"
            docs_path = root / "documents.json"
            manifest_path = root / "index_manifest.json"

            index = faiss.IndexFlatIP(4)
            index.add(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
            faiss.write_index(index, str(index_path))
            docs_path.write_text(json.dumps([
                {"text": "one", "title": "1"},
                {"text": "two", "title": "2"},
            ]), encoding="utf-8")

            kb = KnowledgeBase()
            kb.embedding_model = FakeEmbeddingModel(4)
            kb._embedding_dim = 4
            kb._index_path = index_path
            kb._docs_path = docs_path
            kb._manifest_path = manifest_path

            self.assertFalse(kb.load())
            self.assertIsNone(kb.index)
            self.assertEqual(kb.documents, [])

    def test_load_accepts_legacy_index_without_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = root / "faiss_index.bin"
            docs_path = root / "documents.json"
            manifest_path = root / "index_manifest.json"

            index = faiss.IndexFlatIP(4)
            index.add(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
            faiss.write_index(index, str(index_path))
            docs_path.write_text(json.dumps([
                {"text": "legacy chunk", "title": "legacy"},
            ]), encoding="utf-8")

            kb = KnowledgeBase()
            kb.embedding_model = FakeEmbeddingModel(4)
            kb._embedding_dim = 4
            kb._index_path = index_path
            kb._docs_path = docs_path
            kb._manifest_path = manifest_path

            self.assertTrue(kb.load())
            self.assertTrue(kb.is_ready)
            self.assertEqual(len(kb.documents), 1)

            # 模拟离线环境：嵌入模型不可用时，旧索引仍应能提供 BM25。
            kb._embedding_load_attempted = True
            kb._embedding_load_error = "ImportError: offline"
            results = kb.search("legacy", top_k=1)
            self.assertTrue(results)
            self.assertEqual(results[0]["retrieval_mode"], "bm25")
            self.assertEqual(kb.retrieval_mode, "bm25")


class BuildSafetyTests(unittest.TestCase):
    def test_empty_build_does_not_overwrite_existing_index(self):
        """数据源不可用时应保留已有索引，而不是把它清空后落盘。"""
        kb = KnowledgeBase()
        kb.embedding_model = FakeEmbeddingModel(4)
        kb._embedding_dim = 4
        existing_documents = [{"text": "keep me", "title": "existing"}]
        kb.documents = existing_documents
        kb.index = faiss.IndexFlatIP(4)
        kb.index.add(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        kb._build_bm25_index(existing_documents)

        with patch.object(
            kb, "collect_knowledge_from_opera_data", return_value=[]
        ), patch.object(kb, "save") as save_mock:
            kb.build_index()

        self.assertIs(kb.documents, existing_documents)
        self.assertEqual(kb.index.ntotal, 1)
        save_mock.assert_not_called()

    def test_empty_build_skips_embedding_model_load(self):
        """没有文本时不应为了空建库去加载/下载嵌入模型。"""
        kb = KnowledgeBase()
        with patch.object(
            kb, "collect_knowledge_from_opera_data", return_value=[]
        ), patch.object(
            kb, "_load_embedding_model"
        ) as load_mock, patch.object(kb, "save") as save_mock:
            built = kb.build_index()

        self.assertFalse(built)
        load_mock.assert_not_called()
        save_mock.assert_not_called()

    def test_ensure_ready_does_not_auto_build_by_default(self):
        """请求路径只加载已有索引，不应在 NAS 不可用时主动扫描/建库。"""
        kb = KnowledgeBase()
        with patch.object(kb, "load", return_value=False), patch.object(
            kb, "build_index"
        ) as build_mock:
            self.assertFalse(kb.ensure_ready())

        build_mock.assert_not_called()


class OfflineDegradationTests(unittest.TestCase):
    @staticmethod
    def _make_bm25_kb(documents):
        kb = KnowledgeBase()
        kb.documents = documents
        kb.index = faiss.IndexFlatIP(4)
        kb.index.add(np.asarray(
            [[0.0, 1.0, 0.0, 0.0]] * len(documents),
            dtype=np.float32,
        ))
        kb._build_bm25_index(documents)
        return kb

    def test_search_falls_back_to_bm25_when_embedding_unavailable(self):
        kb = self._make_bm25_kb([{
            "text": "Qiba is a stylized martial posture in Chinese opera.",
            "title": "Qiba",
            "genre": "京剧",
        }])
        kb._embedding_load_attempted = True
        kb._embedding_load_error = "ImportError: offline"

        results = kb.search("什么是起霸？", top_k=3, threshold=0.4)

        self.assertTrue(results)
        self.assertEqual(results[0]["retrieval_mode"], "bm25")
        self.assertFalse(results[0]["dense_available"])
        self.assertEqual(kb.retrieval_mode, "bm25")
        self.assertEqual(kb.embedding_status, "unavailable")
        self.assertAlmostEqual(results[0]["score"], 1.0)

    def test_embedding_load_failure_is_not_retried_per_request(self):
        kb = KnowledgeBase()
        import_calls = []
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                import_calls.append(name)
                raise ImportError("offline model cache missing")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            self.assertFalse(kb._load_embedding_model())
            self.assertFalse(kb._load_embedding_model())

        self.assertEqual(len(import_calls), 1)
        self.assertIn("ImportError", kb.embedding_error)
        self.assertEqual(kb.embedding_status, "unavailable")


class RAGServiceTests(unittest.TestCase):
    @staticmethod
    def _bm25_search_result():
        return {
            "text": "Qiba is a stylized martial posture in Chinese opera.",
            "score": 1.0,
            "dense_score": 0.0,
            "bm25_score": 2.0,
            "matched_by": "bm25",
            "dense_available": False,
            "retrieval_mode": "bm25",
            "title": "Qiba",
            "genre": "京剧",
            "source": "qiba.txt",
        }

    def _answer_with_model(self, search_result, model_answer):
        global_kb = rag_module.knowledge_base
        with patch.object(
            type(global_kb), "is_ready", new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            global_kb, "search", return_value=[search_result],
        ), patch.object(
            type(global_kb), "embedding_status",
            new_callable=PropertyMock, return_value="unavailable",
        ), patch.object(
            type(global_kb), "embedding_error",
            new_callable=PropertyMock, return_value="ImportError: offline",
        ), patch.object(
            type(rag_module.qwen_model), "is_loaded",
            new_callable=PropertyMock, return_value=True,
        ), patch.object(
            rag_module.qwen_model, "chat", return_value=model_answer,
        ):
            return RAGService().answer_question("什么是起霸？", top_k=1)

    def test_query_variants_remove_interrogative_noise(self):
        service = RAGService()
        variants = service._query_variants("请问什么是起霸？")
        self.assertIn("起霸", variants)

    def test_answer_has_citation_ids_and_bounded_context(self):
        search_result = {
            "text": "Qiba is a stylized martial posture in Chinese opera.",
            "score": 0.8,
            "dense_score": 0.5,
            "bm25_score": 2.0,
            "matched_by": "hybrid",
            "title": "Qiba",
            "genre": "京剧",
            "source": "qiba.txt",
        }
        global_kb = rag_module.knowledge_base
        with patch.object(
            type(global_kb), "is_ready", new_callable=PropertyMock,
            return_value=True,
        ), patch.object(
            global_kb, "search", return_value=[search_result],
        ), patch.object(
            type(rag_module.qwen_model), "is_loaded",
            new_callable=PropertyMock, return_value=False,
        ):
            result = RAGService().answer_question("什么是起霸？", top_k=1)

        self.assertEqual(result["source"], "rag")
        self.assertEqual(result["references"][0]["citation_id"], "S1")
        self.assertIn("[S1]", result["answer"])
        self.assertEqual(result["retrieval"]["used"], 1)
        self.assertTrue(result["retrieval"]["hybrid"])
        self.assertEqual(result["retrieval"]["mode"], "hybrid")
        self.assertTrue(result["citation_validation"]["has_valid_citation"])

    def test_answer_removes_fabricated_citation_ids(self):
        result = self._answer_with_model(
            self._bm25_search_result(),
            "起霸是戏曲中的程式动作 [S1]，另见 [S99]。",
        )

        self.assertIn("[S1]", result["answer"])
        self.assertNotIn("[S99]", result["answer"])
        self.assertEqual(
            result["citation_validation"]["cited_citation_ids"], ["S1"]
        )
        self.assertEqual(
            result["citation_validation"]["invalid_citation_ids"], ["S99"]
        )
        self.assertIn("已移除", result["note"])
        # Dense 不可用时，检索元数据必须如实报告 BM25，而不是写死 hybrid。
        self.assertEqual(result["retrieval"]["mode"], "bm25")
        self.assertFalse(result["retrieval"]["hybrid"])
        self.assertEqual(result["retrieval"]["embedding_status"], "unavailable")
        self.assertIn("BM25", result["note"])

    def test_answer_without_valid_citation_is_flagged(self):
        result = self._answer_with_model(
            self._bm25_search_result(),
            "起霸是一种程式动作。",
        )

        self.assertFalse(result["citation_validation"]["has_valid_citation"])
        self.assertFalse(result["citation_validation"]["has_any_citation"])
        self.assertIn("未包含有效资料引用", result["note"])

    def test_retrieval_summary_does_not_treat_bm25_only_as_hybrid(self):
        """Dense 通道有候选不代表最终结果用了 Dense。"""
        result = self._bm25_search_result()
        # 模拟 Dense 通道产生了低分候选，但最终命中的仍是 BM25 结果。
        result["dense_available"] = True

        summary = RAGService._summarize_retrieval([result], [result])

        self.assertEqual(summary["mode"], "bm25")
        self.assertFalse(summary["hybrid"])
        self.assertFalse(summary["dense_available"])
        self.assertTrue(summary["bm25_available"])

    def test_context_budget_truncates_long_references(self):
        service = RAGService()
        results = [{
            "text": "戏" * 10000,
            "score": 0.9,
            "title": "长文档",
            "genre": "京剧",
        }]
        with patch.object(rag_module, "RAG_MAX_CONTEXT_CHARS", 300):
            context, refs = service._format_context(results)

        self.assertEqual(len(refs), 1)
        self.assertLessEqual(len(context), 300)
        self.assertTrue(context.endswith("…"))

    def test_context_respects_requested_top_k(self):
        service = RAGService()
        results = [
            {
                "text": "戏" * 80 + f"第{index}条",
                "score": 0.9 - index * 0.1,
                "title": f"文档{index}",
                "genre": "京剧",
            }
            for index in range(3)
        ]

        context, refs = service._format_context(results, limit=1)

        self.assertEqual(len(refs), 1)
        self.assertIn("文档0", context)
        self.assertNotIn("文档1", context)


class ParameterPropagationTests(unittest.TestCase):
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

    def test_pipeline_passes_rag_top_k_and_threshold(self):
        pipeline = object.__new__(self.pipeline_module.Pipeline)
        pipeline.init_knowledge_base = lambda: None
        with patch.object(
            self.pipeline_module.rag_service, "answer_question",
            return_value={"answer": "ok"},
        ) as answer_mock:
            result = pipeline.ask_question("问题", top_k=3, threshold=0.2)

        self.assertEqual(result, {"answer": "ok"})
        answer_mock.assert_called_once_with("问题", top_k=3, threshold=0.2)

    def test_pipeline_passes_clip_top_k(self):
        pipeline = object.__new__(self.pipeline_module.Pipeline)
        with patch.object(
            self.pipeline_module.clip_retriever, "search",
            return_value=[],
        ) as search_mock:
            pipeline.semantic_search("云手", video_name="video_a", top_k=7)

        search_mock.assert_called_once_with(
            "云手", top_k=7, video_name="video_a"
        )

    def test_pipeline_does_not_eager_load_qwen(self):
        """Web 服务应先可启动，30B 模型在真正问答/分析时再按需加载。"""
        with patch.object(
            self.pipeline_module.Pipeline, "load_qwen_model"
        ) as load_mock:
            self.pipeline_module.Pipeline()

        load_mock.assert_not_called()

    def test_pipeline_knowledge_base_is_load_only(self):
        pipeline = object.__new__(self.pipeline_module.Pipeline)
        pipeline._kb_initialized = False
        with patch.object(
            self.pipeline_module.knowledge_base, "ensure_ready",
            return_value=True,
        ) as ensure_mock:
            pipeline.init_knowledge_base()

        ensure_mock.assert_called_once_with(auto_build=False)

    def test_pipeline_ask_question_enters_qwen_degraded_path(self):
        pipeline = object.__new__(self.pipeline_module.Pipeline)
        pipeline.init_knowledge_base = lambda: None
        with patch.object(
            pipeline, "ensure_qwen", return_value=False
        ) as ensure_mock, patch.object(
            self.pipeline_module.rag_service, "answer_question",
            return_value={"answer": "fallback"},
        ) as answer_mock:
            result = pipeline.ask_question("问题", top_k=2, threshold=0.3)

        ensure_mock.assert_called_once_with()
        answer_mock.assert_called_once_with("问题", top_k=2, threshold=0.3)
        self.assertEqual(result["answer"], "fallback")

    def test_pipeline_status_exposes_retrieval_fields(self):
        pipeline = object.__new__(self.pipeline_module.Pipeline)
        status = pipeline.get_status()

        for key in (
            "knowledge_base_retrieval_mode",
            "knowledge_base_dense_ready",
            "embedding_status",
            "embedding_error",
        ):
            self.assertIn(key, status)


if __name__ == "__main__":
    unittest.main()
