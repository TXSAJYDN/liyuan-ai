"""
Gradio 前端界面：
- 视频上传与分析区
- 结构化结果展示区
- 关键帧语义检索区
- 专业戏曲知识问答区
"""
import os
import sys
import logging
import base64
from pathlib import Path
from typing import Optional, List

import gradio as gr

# ---- 中国风背景 SVG + CSS ----
_BG_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">
  <rect width="400" height="400" fill="transparent"/>
  <g fill="none" stroke="#c41e3a" stroke-width="0.8" opacity="0.06">
    <path d="M50,100 C50,75 70,60 90,60 C90,40 115,30 130,45 C140,30 165,35 165,55 C182,48 198,65 190,85 C205,92 195,115 175,110 C170,125 148,130 138,115 C125,128 100,122 100,105 C82,110 50,102 50,100Z"/>
    <path d="M250,300 C250,275 270,260 290,260 C290,240 315,230 330,245 C340,230 365,235 365,255 C382,248 398,265 390,285 C405,292 395,315 375,310 C370,325 348,330 338,315 C325,328 300,322 300,305 C282,310 250,302 250,300Z"/>
    <path d="M300,70 C300,60 306,56 314,56 C314,46 326,42 332,48 C336,42 346,44 346,52 C352,50 358,56 354,64 C360,68 356,76 348,74 C346,80 336,82 332,76 C326,82 316,78 316,72 C308,74 300,70 300,70Z"/>
    <path d="M80,280 C80,270 86,266 94,266 C94,256 106,252 112,258 C116,252 126,254 126,262 C132,260 138,266 134,274 C140,278 136,286 128,284 C126,290 116,292 112,286 C106,292 96,288 96,282 C88,284 80,280 80,280Z"/>
  </g>
  <g fill="none" stroke="#d4a574" opacity="0.06" stroke-width="0.6">
    <path d="M0,395 Q50,380 100,395 Q150,410 200,395 Q250,380 300,395 Q350,410 400,395"/>
    <path d="M0,5 Q50,-10 100,5 Q150,20 200,5 Q250,-10 300,5 Q350,20 400,5"/>
  </g>
</svg>'''
_BG_B64 = base64.b64encode(_BG_SVG.encode()).decode()
CUSTOM_CSS = f"""
.gradio-container {{
    background:
        url("data:image/svg+xml;base64,{_BG_B64}") repeat,
        linear-gradient(180deg, #faf4ed 0%, #f3e8da 50%, #faf4ed 100%) !important;
    background-attachment: fixed !important;
}}
.block, .form, .panel, .accordion {{
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}}
textarea, input, select, .wrap {{
    background: transparent !important;
    border-bottom: 1px solid rgba(139, 26, 26, 0.2) !important;
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
    border-radius: 0 !important;
}}
.gallery, .gallery .grid-wrap {{
    background: transparent !important;
}}
.tab-nav {{
    background: transparent !important;
}}
h1 {{
    color: #8b1a1a !important;
}}
.tabs > .tab-nav > button.selected {{
    border-color: #c41e3a !important;
    color: #8b1a1a !important;
}}
button.primary {{
    background: linear-gradient(135deg, #c41e3a, #8b1a1a) !important;
    border: none !important;
}}
.icon-col {{
    flex: 0 0 90px !important;
    min-width: 90px !important;
    max-width: 90px !important;
}}
.tab-header-row {{
    justify-content: center !important;
    max-width: 600px !important;
    margin: 0 auto !important;
}}
"""

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- 功能区配图 ----
_STATIC_DIR = Path(__file__).parent / "static"
def _svg_to_html(filename: str, width: int = 120) -> str:
    svg_path = _STATIC_DIR / filename
    if not svg_path.exists():
        return ""
    b64 = base64.b64encode(svg_path.read_bytes()).decode()
    return f'<div style="display:flex;justify-content:center;align-items:center"><img src="data:image/svg+xml;base64,{b64}" width="{width}"/></div>'

_ICON_MASK_HTML = _svg_to_html("icon_mask.svg", 80)
_ICON_FAN_HTML = _svg_to_html("icon_fan.svg", 80)
_ICON_SCROLL_HTML = _svg_to_html("icon_scroll.svg", 80)

from configs.settings import (
    OPERA_DATA_DIR, OPERA_GENRES, GRADIO_PORT, CACHE_DIR,
    RAG_TOP_K, RAG_RELEVANCE_THRESHOLD,
)
from services.pipeline import pipeline
from modules.video_processor import video_processor
from modules.knowledge_base import knowledge_base
from modules.qwen_model import qwen_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_system_status():
    status = pipeline.get_status()
    lines = [
        f"知识库状态: {'已就绪' if status['knowledge_base_ready'] else '未初始化'}"
        f" ({status['knowledge_base_docs']} 个文档)",
        f"CLIP索引: {'已建立' if status['clip_index_ready'] else '待建立'}"
        f" ({status['clip_index_size']} 个向量)",
        f"千问模型: {'已加载' if status['qwen_model_loaded'] else '未加载（按需启动）'}",
    ]
    return "\n".join(lines)


def init_kb():
    try:
        pipeline.init_knowledge_base()
        return "知识库初始化完成！"
    except Exception as e:
        return f"知识库初始化失败: {e}"


def load_qwen():
    try:
        pipeline.load_qwen_model()
        return "千问模型加载完成！"
    except Exception as e:
        return f"千问模型加载失败: {e}"


def list_genre_videos(genre_key):
    if not genre_key:
        return gr.update(choices=[], value=None)
    videos = video_processor.list_opera_videos(genre_key)
    choices = [v["name"] for v in videos]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def list_indexed_videos():
    """扫描缓存目录，返回已建立 CLIP 索引的视频名列表"""
    index_dir = CACHE_DIR / "clip_indices"
    if not index_dir.exists():
        return []
    return sorted(
        p.stem for p in index_dir.glob("*.index")
    )


def strip_markdown(text: str) -> str:
    """去除文本中的 Markdown 粗体标记"""
    return text.replace("**", "")


def format_analysis(raw_analysis: str) -> str:
    """将模型输出的 JSON 分析结果格式化为连贯的文字描述"""
    import json as _json
    import re
    # 尝试从 markdown 代码块中提取 JSON
    m = re.search(r'```(?:json)?\s*(.+?)\s*```', raw_analysis, re.DOTALL)
    if not m:
        return raw_analysis
    try:
        data = _json.loads(m.group(1))
    except _json.JSONDecodeError:
        return raw_analysis
    # 字段映射：JSON key -> 显示标签
    field_labels = [
        ("剧种剧目推测", "剧种与剧目"),
        ("核心行当", "核心行当"),
        ("关键动作程式", "关键动作程式"),
        ("唱腔板式推测", "唱腔板式"),
        ("唱腔 板式推测", "唱腔板式"),
        ("情感表达", "情感表达"),
        ("情感 表达", "情感表达"),
        ("道具服饰细节", "道具与服饰"),
        (" 道具服饰细节", "道具与服饰"),
        ("综合描述", None),
    ]
    parts = []
    for key, label in field_labels:
        val = data.get(key)
        if not val:
            continue
        if label is None:
            parts.append(f"\n{val}")
        else:
            parts.append(f"【{label}】{val}")
    if not parts:
        return strip_markdown(raw_analysis)
    return strip_markdown("\n".join(parts))


def process_and_analyze_uploaded(video_file, progress=gr.Progress()):
    if video_file is None:
        return "请先上传视频文件", [], "", gr.update()
    try:
        video_path = video_file
        progress(0.1, desc="正在处理视频（切片 + 抽帧）...")
        result = pipeline.process_uploaded_video(video_path)
        keyframe_paths = [kf["path"] for kf in result["keyframes"]]
        progress(0.5, desc="正在AI结构化分析...")
        analysis = pipeline.analyze_video(video_path, keyframe_paths=keyframe_paths)
        thumbnails = analysis["sampled_frames"]
        progress(0.9, desc="生成结果...")
        info = (
            f"视频信息: {result['video_info']['width']}x{result['video_info']['height']}, "
            f"{result['video_info']['fps']:.1f}fps, "
            f"时长 {result['video_info']['duration']:.1f}s\n"
            f"切片数: {len(result['segments'])}\n"
            f"关键帧数: {len(keyframe_paths)}（分析用: {len(thumbnails)}）\n"
            f"CLIP索引: {'已建立' if result.get('clip_index_ready') else '未建立'}"
        )
        video_name = Path(video_path).stem
        indexed = list_indexed_videos()
        progress(1.0, desc="完成")
        return (
            info, thumbnails, format_analysis(analysis["analysis"]),
            gr.update(choices=indexed, value=video_name),
        )
    except Exception as e:
        logger.error(f"视频处理失败: {e}", exc_info=True)
        return f"处理失败: {e}", [], "", gr.update()


def process_and_analyze_opera(genre_key, video_name, progress=gr.Progress()):
    if not genre_key or not video_name:
        return "请选择剧种和视频", [], "", gr.update()
    try:
        progress(0.1, desc="正在处理视频（切片 + 抽帧）...")
        result = pipeline.process_opera_video(genre_key, video_name)
        keyframe_paths = [kf["path"] for kf in result["keyframes"]]
        progress(0.5, desc="正在AI结构化分析...")
        analysis = pipeline.analyze_video(result["video_path"], keyframe_paths=keyframe_paths)
        thumbnails = analysis["sampled_frames"]
        progress(0.9, desc="生成结果...")
        info = (
            f"视频信息: {result['video_info']['width']}x{result['video_info']['height']}, "
            f"{result['video_info']['fps']:.1f}fps, "
            f"时长 {result['video_info']['duration']:.1f}s\n"
            f"切片数: {len(result['segments'])}\n"
            f"关键帧数: {len(keyframe_paths)}（分析用: {len(thumbnails)}）\n"
            f"CLIP索引: {'已建立' if result.get('clip_index_ready') else '未建立'}"
        )
        vname = Path(result["video_path"]).stem
        indexed = list_indexed_videos()
        progress(1.0, desc="完成")
        return (
            info, thumbnails, format_analysis(analysis["analysis"]),
            gr.update(choices=indexed, value=vname),
        )
    except Exception as e:
        logger.error(f"视频处理失败: {e}", exc_info=True)
        return f"处理失败: {e}", [], "", gr.update()


def do_semantic_search(query, video_name):
    if not query:
        return "请输入检索内容", []
    try:
        results = pipeline.semantic_search(query, video_name=video_name or None)
        if not results:
            return "未找到匹配的关键帧。请先处理视频建立索引。", []
        images = [r["path"] for r in results]
        info_lines = [f"检索到 {len(results)} 个匹配关键帧："]
        for i, r in enumerate(results):
            line = f"  {i+1}. 时间戳: {r.get('timestamp', 'N/A')}s, 相似度: {r['score']:.4f}"
            if "explanation" in r:
                line += f"\n     解释: {r['explanation']}"
            info_lines.append(line)
        return "\n".join(info_lines), images
    except Exception as e:
        logger.error(f"语义检索失败: {e}", exc_info=True)
        return f"检索失败: {e}", []


def do_qa_stream(question):
    """流式问答：逐字输出 Qwen 回答"""
    if not question:
        yield "请输入问题"
        return
    try:
        pipeline.init_knowledge_base()
        search_results = knowledge_base.search(
            question, top_k=RAG_TOP_K, threshold=RAG_RELEVANCE_THRESHOLD
        )
        if not qwen_model.is_loaded:
            result = pipeline.ask_question(question)
            yield result["answer"]
            return
        system_prompt = (
            "你是一位精通中国戏曲的专家，拥有丰富的戏曲知识。"
            "请根据提供的参考资料（如有）回答用户关于戏曲的问题。"
            "回答要准确、专业、通俗易懂。"
        )
        if search_results:
            knowledge_context = "\n\n".join([
                f"[来源: {r.get('title', '未知')} ({r.get('genre', '')})] {r['text']}"
                for r in search_results
            ])
            prompt = (
                f"参考资料：\n{knowledge_context}\n\n"
                f"用户问题：{question}\n\n"
                "请结合以上参考资料和你的专业知识，直接回答用户的问题。"
                "回答应融会贯通，无需区分内容来源。"
            )
        else:
            prompt = (
                f"用户问题：{question}\n\n"
                "请根据你的戏曲专业知识回答。"
            )
        accumulated = ""
        for token in qwen_model.chat_stream(prompt, system_prompt):
            accumulated += token
            yield strip_markdown(accumulated)
        if search_results:
            refs = "\n\n---\n参考来源：\n"
            for i, ref in enumerate(search_results):
                refs += (
                    f"  {i+1}. [{ref.get('title', '未知')}] "
                    f"({ref.get('genre', '')}) "
                    f"相关度: {ref['score']:.4f}\n"
                )
            yield strip_markdown(accumulated) + refs
    except Exception as e:
        logger.error(f"问答失败: {e}", exc_info=True)
        yield f"问答失败: {e}"


def export_report(info_text, analysis_text):
    """导出分析报告为可下载文件"""
    if not analysis_text:
        return gr.update(visible=False)
    from datetime import datetime
    report_dir = Path("data/cache/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = report_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("梨园AI - 戏曲视频结构化分析报告\n")
        f.write("=" * 40 + "\n\n")
        if info_text:
            f.write(f"{info_text}\n\n")
        f.write(f"分析结果：\n{analysis_text}\n")
    return gr.update(value=str(filepath), visible=True)


def create_ui():
    genre_choices = list(OPERA_GENRES.keys())

    with gr.Blocks(
        title="梨园AI - 智能戏曲分析与交互平台",
    ) as demo:

        gr.Markdown("# 梨园AI - 智能戏曲分析与交互平台")
        gr.Markdown("基于多模态大模型的戏曲视频结构化分析、语义检索与专业问答")

        with gr.Accordion("使用指引（点击展开）", open=True):
            gr.Markdown(
                '**准备：** 展开下方「系统状态与初始化」，点击「加载千问模型」，等待模型加载完成后再使用各功能。\n\n'
                '**第一步：** 在「视频结构化分析」中选择一个戏曲视频，点击"开始分析"。其中「上传视频」用于上传你自己的视频文件，「从数据库选择」可直接选用系统预置的各剧种视频素材。AI 将自动提取关键帧并生成结构化分析报告。\n\n'
                '**第二步：** 分析完成后，切换到「按义寻画」，用自然语言描述你想找的画面（如"水袖动作"），AI 会从视频中精准定位匹配帧。\n\n'
                '**第三步：** 在「戏曲知识问答」中提出任何戏曲相关问题，AI 结合专业知识库为你解答。'
            )

        with gr.Accordion("系统状态与初始化", open=False):
            status_text = gr.Textbox(label="系统状态", lines=3, interactive=False)
            with gr.Row():
                btn_refresh = gr.Button("刷新状态")
                btn_init_kb = gr.Button("初始化知识库")
                btn_load_qwen = gr.Button("加载千问模型")
            init_result = gr.Textbox(label="操作结果", interactive=False)

        with gr.Tabs():
            with gr.Tab("视频结构化分析"):
                with gr.Row(elem_classes=["tab-header-row"]):
                    gr.HTML(_ICON_MASK_HTML, elem_classes=["icon-col"])
                    gr.Markdown("### 上传或选择戏曲视频\nAI将自动分析并生成结构化总结")
                with gr.Tabs():
                    with gr.Tab("上传视频"):
                        upload_video = gr.Video(label="上传戏曲视频")
                        btn_analyze_upload = gr.Button("开始分析", variant="primary")
                    with gr.Tab("从数据库选择"):
                        with gr.Row():
                            genre_dropdown = gr.Dropdown(
                                choices=genre_choices, label="选择剧种",
                                info="选择戏曲剧种分类"
                            )
                            video_dropdown = gr.Dropdown(
                                choices=[], label="选择视频",
                                info="选择具体视频"
                            )
                        btn_analyze_opera = gr.Button("开始分析", variant="primary")
                with gr.Row():
                    with gr.Column(scale=1):
                        analysis_info = gr.Textbox(label="处理信息", lines=5, interactive=False)
                    with gr.Column(scale=1):
                        analysis_result = gr.Textbox(label="结构化分析结果", lines=15, interactive=False)
                keyframe_gallery = gr.Gallery(label="关键帧预览", columns=5, height=300)
                with gr.Row():
                    btn_export = gr.Button("导出分析报告")
                    export_file = gr.File(label="下载报告", visible=False)

            with gr.Tab("按义寻画（语义检索）"):
                with gr.Row(elem_classes=["tab-header-row"]):
                    gr.HTML(_ICON_FAN_HTML, elem_classes=["icon-col"])
                    gr.Markdown('### 用自然语言描述你想找的画面\nAI帮你从视频中精准定位。示例："甩水袖"、"起霸动作"、"人物亮相"、"脸谱特写"')
                with gr.Row():
                    search_query = gr.Textbox(
                        label="输入描述", placeholder="描述你想找的戏曲画面...", scale=3
                    )
                    search_video_dropdown = gr.Dropdown(
                        label="选择视频",
                        choices=list_indexed_videos(),
                        value=None,
                        allow_custom_value=True,
                        info="选择已索引的视频，或留空搜索当前视频",
                        scale=1,
                    )
                btn_search = gr.Button("搜索", variant="primary")
                search_info = gr.Textbox(label="检索结果", lines=8, interactive=False)
                search_gallery = gr.Gallery(label="匹配的关键帧", columns=5, height=400)

            with gr.Tab("戏曲知识问答"):
                with gr.Row(elem_classes=["tab-header-row"]):
                    gr.HTML(_ICON_SCROLL_HTML, elem_classes=["icon-col"])
                    gr.Markdown('### 提出你的戏曲问题，AI基于专业知识库为你解答\n示例："西皮和二黄有什么区别？"、"什么是起霸？"、"梆子戏有哪些代表剧目？"')
                qa_input = gr.Textbox(
                    label="输入问题", placeholder="请输入你的戏曲相关问题...", lines=2
                )
                btn_qa = gr.Button("提问", variant="primary")
                qa_output = gr.Textbox(label="回答", lines=12, interactive=False)

        gr.Markdown(
            "---\n"
            "**梨园AI** | 基于多模态大模型的智能戏曲分析与交互平台 | "
            "技术栈：Qwen3-Omni + CLIP + FAISS + FastAPI + Gradio"
        )

        # ---- 事件绑定（所有组件定义完毕后统一绑定） ----
        btn_refresh.click(fn=get_system_status, outputs=status_text)
        btn_init_kb.click(fn=init_kb, outputs=init_result)
        btn_load_qwen.click(fn=load_qwen, outputs=init_result)

        genre_dropdown.change(
            fn=list_genre_videos, inputs=genre_dropdown, outputs=video_dropdown
        )
        btn_analyze_upload.click(
            fn=process_and_analyze_uploaded,
            inputs=[upload_video],
            outputs=[analysis_info, keyframe_gallery, analysis_result, search_video_dropdown],
        )
        btn_analyze_opera.click(
            fn=process_and_analyze_opera,
            inputs=[genre_dropdown, video_dropdown],
            outputs=[analysis_info, keyframe_gallery, analysis_result, search_video_dropdown],
        )
        btn_export.click(
            fn=export_report,
            inputs=[analysis_info, analysis_result],
            outputs=[export_file],
        )
        btn_search.click(
            fn=do_semantic_search,
            inputs=[search_query, search_video_dropdown],
            outputs=[search_info, search_gallery],
        )
        btn_qa.click(fn=do_qa_stream, inputs=[qa_input], outputs=[qa_output])

        demo.load(fn=get_system_status, outputs=status_text)

    return demo


def main():
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
