# Liyuan AI — Intelligent Opera Analysis and Interaction Platform

A multimodal system for structured analysis of opera videos, semantic retrieval, and expert Q&A.

## Project Structure

```
liyuan-ai/
├── app/                    # Application layer
│   ├── api.py             # FastAPI backend API
│   ├── gradio_ui.py       # Gradio frontend UI
│   └── static/            # Static assets (UI icons, SVGs, etc.)
├── modules/               # AI core modules
│   ├── video_processor.py # Video processing (FFmpeg slicing + OpenCV keyframe extraction)
│   ├── knowledge_base.py  # Knowledge base (FAISS vector retrieval)
│   ├── clip_retriever.py  # CLIP semantic image retrieval
│   └── qwen_model.py      # Qwen multimodal model wrapper
├── services/              # Service layer
│   ├── pipeline.py        # Core business pipeline
│   └── rag_service.py     # RAG-enhanced Q&A service
├── configs/               # Configuration
│   └── settings.py        # Global settings
├── knowledge_base/        # Knowledge base storage
│   ├── vector_store/      # FAISS index
│   └── raw_texts/         # Source knowledge texts
├── data/                  # Local data
│   ├── uploads/           # Uploaded videos
│   ├── processed/         # Processed video segments
│   ├── keyframes/         # Extracted keyframes
│   └── cache/             # Cache (CLIP indices, etc.)
├── run_api.py             # Start backend service
├── run_ui.py              # Start frontend UI
├── build_knowledge_base.py # Build knowledge base
└── requirements.txt       # Dependencies
```

## Environment Setup

```bash
conda activate liyuan
pip install -r requirements.txt
```

## Core Features

### Feature 1: Structured Summary of Opera Videos
Upload or select an opera video → automatic slicing and keyframe extraction → RAG retrieves domain knowledge → Qwen model generates structured analysis.

### Feature 2: Search by Meaning (Semantic Retrieval)
Input natural language description → CLIP encodes text → compute similarity with keyframe vectors → return the best-matching frames.

### Feature 3: Expert Opera Knowledge Q&A
RAG retrieves from the knowledge base for professional answers; the Qwen model provides general fallback responses.

## Usage

### Step 1: Build the Knowledge Base

```bash
conda activate liyuan
cd /path/to/liyuan-ai
python build_knowledge_base.py
```

### Step 2: Start Services

**Option A: Launch Gradio UI (recommended for demo)**
```bash
python run_ui.py
# Visit http://localhost:7860
```

**Option B: Launch FastAPI Backend**
```bash
python run_api.py
# Visit http://localhost:8000/docs for API docs
```

### Step 3: Load the Qwen Model On Demand
In the Gradio UI, click "Load Qwen model", or call the API:
```bash
curl -X POST http://localhost:8000/api/init/qwen_model
```

## Key Configuration

| Setting            | Value                                                | Description                               |
|--------------------|------------------------------------------------------|-------------------------------------------|
| Opera data         | `/srv/nas_data/opera`(The server)                    | Read-only access                          |
| Qwen model         | `/home/shq/data/models/Qwen3-omni-30B-A3B-Instruct`  | Load on demand                            |
| Embedding model    | `paraphrase-multilingual-MiniLM-L12-v2`              | Knowledge base vectorization              |
| CLIP model         | `ViT-B-16 (openai)`                                  | Semantic image retrieval                  |
| Vector database    | FAISS (IndexFlatIP)                                  | KB + CLIP indices                         |

## Tech Stack

- Backend: FastAPI + Python
- Frontend: Gradio
- Video Processing: FFmpeg + OpenCV
- Multimodal Model: Qwen3-Omni-30B
- Semantic Retrieval: OpenCLIP (ViT-B-16)
- Knowledge Base: FAISS + Sentence-Transformers
- Q&A: RAG (Retrieval-Augmented Generation)
