

# Multimodal RAG with Universal Evidence Citing

A RAG system that handles ambiguity, adapts retrieval, and acknowledges uncertainty.

## Features

- **Multimodal Ingestion**: PDF, DOCX, Markdown, Images, Audio, Video
- **Cross-Modal Retrieval**: Search across all modalities
- **Universal Citations**: Every answer grounded in evidence
- **Conflict Detection**: LLM-based contradiction identification
- **Hallucination Guard**: Refuses when confidence is low
- **Obsidian Export**: Linked notes for audit trail

## Quick Start

### 1. Prerequisites

- Python 3.10+
- FFmpeg (for audio/video processing)
- OpenRouter API key

### 2. Install FFmpeg

**Windows:**
```bash
choco install ffmpeg
# or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 3. Setup

```bash

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and add your OPENROUTER_API_KEY OR USE LOCAL LLM
```

### 4. Run

```bash
# Start backend
cd backend
uvicorn main:app --reload --port 8000

# Open frontend
# Open frontend/index.html in browser
# Or serve it: python -m http.server 3000 --directory frontend
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ingest` | POST | Upload and index a file |
| `/query` | POST | Query the knowledge base |
| `/evidence/{id}` | GET | Get raw evidence |
| `/export/obsidian` | POST | Export to Obsidian |

## Supported File Types

| Type | Extensions |
|------|------------|
| Documents | `.pdf`, `.docx`, `.pptx`, `.md`, `.txt`, `.html` |
| Images | `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif` |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` |
| Video | `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm` |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Ingest    │ ──▶ │   LanceDB   │ ◀── │  Retrieve   │
│  (Parsers)  │     │  (Vectors)  │     │  (Search)   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Conflict   │ ◀── │  Reasoning  │ ──▶ │    Guard    │
│  Detection  │     │             │     │  (Refuse?)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  OpenRouter │ ──▶ │   Answer    │ ──▶ │  Citations  │
│    (LLM)    │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | - | Required. Get from openrouter.ai |
| `PRIMARY_MODEL` | gemini flash 2.0 lite | Main LLM for answers |
| `FAST_MODEL` | gpt oss 120B | Fast model for conflict detection |
| `MAX_VIDEO_SIZE_MB` | 100 | Max video file size |
| `MAX_VIDEO_DURATION_SEC` | 600 | Max video length (10 min) |

## Uncertainty Calculation

Confidence score based on 4 factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Embedding Similarity | 35% | Vector search score |
| OCR/ASR Confidence | 20% | Extraction quality |
| Modality Agreement | 25% | Cross-modal consistency |
| Source Count | 20% | Number of sources |

**Thresholds:**
- Below 0.4 → Refuse to answer
- Below 0.6 → Show warning

