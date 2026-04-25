# 梨园知见 - 融合视觉语义检索与知识增强生成的戏曲视频智能解析平台

基于多模态大模型的戏曲视频结构化分析、语义检索与专业问答系统。

## 项目架构

```
liyuan-ai2/
├── app/                    # 应用层
│   ├── api.py             # FastAPI 后端 JSON 接口
│   ├── web.py             # Jinja2 页面路由
│   ├── gradio_ui.py       # Gradio 前端界面（可选）
│   ├── templates/         # HTML 页面模板
│   │   ├── base.html      # 公共基础模板
│   │   ├── index.html     # 首页
│   │   ├── analyze.html   # 视频分析页
│   │   ├── search.html    # 语义检索页
│   │   └── qa.html        # 知识问答页
│   └── static/            # 前端静态资源
│       ├── css/main.css   # 全局样式
│       ├── js/main.js     # 全局脚本
│       ├── icons/         # 图标资源
│       ├── images/        # 图片资源
│       └── *.svg          # SVG 图标
├── modules/               # AI 核心模块
│   ├── video_processor.py # 视频处理（FFmpeg切片 + OpenCV抽帧）
│   ├── knowledge_base.py  # 知识库（FAISS向量检索）
│   ├── clip_retriever.py  # CLIP语义图像检索
│   └── qwen_model.py      # 千问多模态模型封装
├── services/              # 业务服务层
│   ├── pipeline.py        # 核心业务流水线
│   └── rag_service.py     # RAG增强问答服务
├── configs/               # 配置
│   └── settings.py        # 全局配置
├── knowledge_base/        # 知识库存储
│   ├── vector_store/      # FAISS 索引
│   └── raw_texts/         # 原始知识文本
├── data/                  # 本地数据
│   ├── uploads/           # 上传的视频
│   ├── processed/         # 处理后的视频片段
│   ├── keyframes/         # 抽取的关键帧
│   └── cache/             # 缓存（分析结果、CLIP索引等）
├── run_api.py             # 启动后端 API 服务（端口 8000）
├── run_ui.py              # 启动前端界面服务（端口 7860）
├── build_knowledge_base.py # 构建知识库
└── requirements.txt       # 依赖列表
```

## 环境配置

```bash
conda activate liyuan
pip install -r requirements.txt
```

## 三大核心功能

### 功能一：戏曲视频结构化总结
上传或选择戏曲视频 → 自动切片抽帧 → RAG 检索专业知识 → 千问模型生成结构化分析

### 功能二：按义寻画（语义检索）
用户输入自然语言描述 → CLIP 文本编码 → 与关键帧图像向量计算相似度 → 返回最匹配画面

### 功能三：专业戏曲知识问答
RAG 优先检索知识库生成专业答案，通用兜底调用千问通用能力回答

## 使用方式

### 第一步：构建知识库
```bash
conda activate liyuan
python build_knowledge_base.py
```

### 第二步：启动服务

**方式A：启动 Web 前端（FastAPI + Jinja2，推荐演示用）**
```bash
python run_ui.py
# 访问 http://localhost:7860
```
> 启动时自动加载千问模型（Qwen3-Omni），无需手动操作。

**方式B：启动 FastAPI 后端（纯 API）**
```bash
python run_api.py
# 访问 http://localhost:8000/docs 查看 API 文档
```

**方式C：启动 Gradio 前端（可选）**
```bash
python -c "from app.gradio_ui import demo; demo.launch()"
# 访问 http://localhost:7860
```

## 关键配置

| 配置项 | 值 | 说明 |
|--------|------|------|
| 戏曲数据 | `/srv/nas_data/opera` | 只读访问，不修改原始数据 |
| 千问模型 | `/home/shq/data/models/Qwen3-omni-30B-A3B-Instruct` | 启动时自动加载 |
| 嵌入模型 | `paraphrase-multilingual-MiniLM-L12-v2` | 知识库向量化 |
| CLIP 模型 | `ViT-B-16 (openai)` | 语义图像检索 |
| 向量数据库 | FAISS (IndexFlatIP) | 知识库 + CLIP 索引 |

## 技术栈

- **后端**: FastAPI + Python
- **前端**: Jinja2 模板 + 原生 JavaScript（Gradio 可选）
- **视频处理**: FFmpeg + OpenCV
- **多模态模型**: Qwen3-Omni-30B
- **语义检索**: OpenCLIP (ViT-B-16)
- **知识库**: FAISS + Sentence-Transformers
- **问答**: RAG（检索增强生成）
