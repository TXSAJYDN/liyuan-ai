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
from pathlib import Path
from typing import Optional, List

import gradio as gr

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import OPERA_DATA_DIR, OPERA_GENRES, GRADIO_PORT
from services.pipeline import pipeline
from modules.video_processor import video_processor

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


def process_and_analyze_uploaded(video_file):
    if video_file is None:
        return "请先上传视频文件", [], ""
    try:
        video_path = video_file
        result = pipeline.process_uploaded_video(video_path)
        keyframe_paths = [kf["path"] for kf in result["keyframes"]]
        analysis = pipeline.analyze_video(video_path, keyframe_paths=keyframe_paths)
        thumbnails = analysis["sampled_frames"]
        info = (
            f"视频信息: {result['video_info']['width']}x{result['video_info']['height']}, "
            f"{result['video_info']['fps']:.1f}fps, "
            f"时长 {result['video_info']['duration']:.1f}s\n"
            f"切片数: {len(result['segments'])}\n"
            f"关键帧数: {len(keyframe_paths)}（分析用: {len(thumbnails)}）\n"
            f"CLIP索引: {'已建立' if result.get('clip_index_ready') else '未建立'}"
        )
        return info, thumbnails, analysis["analysis"]
    except Exception as e:
        logger.error(f"视频处理失败: {e}", exc_info=True)
        return f"处理失败: {e}", [], ""


def process_and_analyze_opera(genre_key, video_name):
    if not genre_key or not video_name:
        return "请选择剧种和视频", [], ""
    try:
        result = pipeline.process_opera_video(genre_key, video_name)
        keyframe_paths = [kf["path"] for kf in result["keyframes"]]
        analysis = pipeline.analyze_video(result["video_path"], keyframe_paths=keyframe_paths)
        thumbnails = analysis["sampled_frames"]
        info = (
            f"视频信息: {result['video_info']['width']}x{result['video_info']['height']}, "
            f"{result['video_info']['fps']:.1f}fps, "
            f"时长 {result['video_info']['duration']:.1f}s\n"
            f"切片数: {len(result['segments'])}\n"
            f"关键帧数: {len(keyframe_paths)}（分析用: {len(thumbnails)}）\n"
            f"CLIP索引: {'已建立' if result.get('clip_index_ready') else '未建立'}"
        )
        return info, thumbnails, analysis["analysis"]
    except Exception as e:
        logger.error(f"视频处理失败: {e}", exc_info=True)
        return f"处理失败: {e}", [], ""


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


def do_qa(question):
    if not question:
        return "请输入问题"
    try:
        result = pipeline.ask_question(question)
        answer = result["answer"]
        source_tag = "知识增强回答" if result["source"] == "rag" else "通用生成回答"
        note = result.get("note", "")
        output = f"**{source_tag}**\n\n{answer}"
        if note:
            output += f"\n\n---\n{note}"
        if result.get("references"):
            output += "\n\n---\n**参考来源：**\n"
            for i, ref in enumerate(result["references"]):
                output += (
                    f"  {i+1}. [{ref.get('title', '未知')}] "
                    f"({ref.get('genre', '')}) "
                    f"相关度: {ref['score']:.4f}\n"
                )
        return output
    except Exception as e:
        logger.error(f"问答失败: {e}", exc_info=True)
        return f"问答失败: {e}"


def create_ui():
    genre_choices = list(OPERA_GENRES.keys())

    with gr.Blocks(
        title="梨园AI - 智能戏曲分析与交互平台",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("# 梨园AI - 智能戏曲分析与交互平台")
        gr.Markdown("基于多模态大模型的戏曲视频结构化分析、语义检索与专业问答")

        with gr.Accordion("系统状态与初始化", open=False):
            status_text = gr.Textbox(label="系统状态", lines=3, interactive=False)
            with gr.Row():
                btn_refresh = gr.Button("刷新状态")
                btn_init_kb = gr.Button("初始化知识库")
                btn_load_qwen = gr.Button("加载千问模型")
            init_result = gr.Textbox(label="操作结果", interactive=False)
            btn_refresh.click(fn=get_system_status, outputs=status_text)
            btn_init_kb.click(fn=init_kb, outputs=init_result)
            btn_load_qwen.click(fn=load_qwen, outputs=init_result)

        with gr.Tabs():
            with gr.Tab("视频结构化分析"):
                gr.Markdown("### 上传或选择戏曲视频，AI将自动分析并生成结构化总结")
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
                        genre_dropdown.change(
                            fn=list_genre_videos,
                            inputs=genre_dropdown,
                            outputs=video_dropdown
                        )
                with gr.Row():
                    with gr.Column(scale=1):
                        analysis_info = gr.Textbox(label="处理信息", lines=5, interactive=False)
                    with gr.Column(scale=1):
                        analysis_result = gr.Textbox(label="结构化分析结果", lines=15, interactive=False)
                keyframe_gallery = gr.Gallery(label="关键帧预览", columns=5, height=300)
                btn_analyze_upload.click(
                    fn=process_and_analyze_uploaded,
                    inputs=[upload_video],
                    outputs=[analysis_info, keyframe_gallery, analysis_result]
                )
                btn_analyze_opera.click(
                    fn=process_and_analyze_opera,
                    inputs=[genre_dropdown, video_dropdown],
                    outputs=[analysis_info, keyframe_gallery, analysis_result]
                )

            with gr.Tab("按义寻画（语义检索）"):
                gr.Markdown("### 用自然语言描述你想找的画面，AI帮你从视频中精准定位")
                gr.Markdown('示例："主角甩水袖的瞬间"、"起霸动作"、"人物亮相"、"脸谱特写"')
                with gr.Row():
                    search_query = gr.Textbox(
                        label="输入描述", placeholder="描述你想找的戏曲画面...", scale=3
                    )
                    search_video_name = gr.Textbox(
                        label="视频名称（可选）",
                        placeholder="留空则搜索当前已索引视频", scale=1
                    )
                btn_search = gr.Button("搜索", variant="primary")
                search_info = gr.Textbox(label="检索结果", lines=8, interactive=False)
                search_gallery = gr.Gallery(label="匹配的关键帧", columns=5, height=400)
                btn_search.click(
                    fn=do_semantic_search,
                    inputs=[search_query, search_video_name],
                    outputs=[search_info, search_gallery]
                )

            with gr.Tab("戏曲知识问答"):
                gr.Markdown("### 提出你的戏曲问题，AI基于专业知识库为你解答")
                gr.Markdown(
                    '示例："京剧中的西皮和二黄有什么区别？"、'
                    '"什么是起霸？"、"梆子戏有哪些代表剧目？"'
                )
                qa_input = gr.Textbox(
                    label="输入问题", placeholder="请输入你的戏曲相关问题...", lines=2
                )
                btn_qa = gr.Button("提问", variant="primary")
                qa_output = gr.Textbox(label="回答", lines=12, interactive=False)
                btn_qa.click(fn=do_qa, inputs=[qa_input], outputs=[qa_output])

        gr.Markdown(
            "---\n"
            "**梨园AI** | 基于多模态大模型的智能戏曲分析与交互平台 | "
            "技术栈：Qwen3-Omni + CLIP + FAISS + FastAPI + Gradio"
        )
        demo.load(fn=get_system_status, outputs=status_text)

    return demo


def main():
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
