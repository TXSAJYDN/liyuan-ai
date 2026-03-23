"""
独立脚本：构建戏曲知识库
从 /srv/nas_data/opera 中的 txt 文件构建向量化知识库
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

from modules.knowledge_base import knowledge_base
from configs.settings import OPERA_DATA_DIR, OPERA_GENRES

def main():
    print("=" * 60)
    print("梨园AI - 戏曲知识库构建工具")
    print("=" * 60)
    print(f"\n数据源目录: {OPERA_DATA_DIR}")
    print(f"剧种分类: {', '.join(f'{k}({v})' for k, v in OPERA_GENRES.items())}")

    docs = knowledge_base.collect_knowledge_from_opera_data()
    print(f"\n收集到 {len(docs)} 个知识文档")
    for genre in sorted(set(d["genre"] for d in docs)):
        count = sum(1 for d in docs if d["genre"] == genre)
        print(f"  - {genre}: {count} 个文档")

    print("\n开始构建向量索引...")
    knowledge_base.build_index()

    print(f"\n知识库构建完成！")
    print(f"   文档块数: {len(knowledge_base.documents)}")
    print(f"   向量数量: {knowledge_base.index.ntotal}")
    print(f"   索引已保存到: {knowledge_base._index_path}")

    print("\n--- 测试检索 ---")
    test_queries = ["京剧行当", "梆子戏", "水袖动作"]
    for q in test_queries:
        results = knowledge_base.search(q, top_k=2)
        print(f"\n查询: '{q}'")
        for r in results:
            print(f"  [{r['score']:.4f}] [{r.get('title', '')}] {r['text'][:80]}...")


if __name__ == "__main__":
    main()
