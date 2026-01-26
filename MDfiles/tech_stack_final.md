# CHAKRAVYUH - Minimal Tech Stack (Final)

## Removed (Unnecessary)
- ~~TypeScript~~ → Plain JavaScript
- ~~Video.js~~ → Native `<video>`
- ~~D3.js~~ → Simple HTML/CSS
- ~~Tailwind~~ → Plain CSS
- ~~NLI models (DeBERTa)~~ → LLM-based conflict detection
- ~~Camelot (table extraction)~~ → Docling handles it
- ~~Word-level timestamps~~ → Segment timestamps only
- ~~Daily summary notes~~ → Just query + evidence notes
- ~~Local 7B models~~ → OpenRouter only

---

## Final Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                     CHAKRAVYUH (MINIMAL)                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION                                │
├───────────────┬───────────────┬───────────────┬─────────────────┤
│  DOCUMENTS    │    IMAGES     │    AUDIO      │     VIDEO       │
│  ───────────  │  ───────────  │  ───────────  │  ─────────────  │
│  Docling      │  Pillow       │  Whisper      │  MoviePy        │
│  (PDF, DOCX,  │  EasyOCR      │  (segments)   │  (frames)       │
│   MD, HTML)   │               │               │  yt-dlp         │
└───────────────┴───────────────┴───────────────┴─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EMBEDDINGS                                 │
│            sentence-transformers + CLIP                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LANCEDB                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RAG ENGINE                                  │
├───────────────┬───────────────┬─────────────────────────────────┤
│   RETRIEVER   │   CONFLICT    │     HALLUCINATION GUARD         │
│  LanceDB      │  (via LLM)    │   confidence threshold          │
└───────────────┴───────────────┴─────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OPENROUTER                                 │
│              claude-3.5-sonnet / gpt-4o-mini                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FRONTEND                                  │
│         HTML + CSS + JavaScript (no frameworks)                 │
│         PDF.js | <img> | <audio> | <video>                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OBSIDIAN EXPORT                              │
│              Query notes + Evidence notes only                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Libraries (Minimal)

| Purpose | Library |
|---------|---------|
| Documents | `docling` |
| Images | `Pillow`, `easyocr` |
| Audio | `openai-whisper` |
| Video | `moviepy`, `opencv-python`, `yt-dlp` |
| Embeddings | `sentence-transformers`, `transformers` (CLIP) |
| Vector DB | `lancedb` |
| API | `fastapi`, `uvicorn` |
| HTTP | `httpx` |
| Frontend | HTML + CSS + JS (vanilla) |
| PDF Viewer | `pdf.js` (CDN) |

---

## Conflict Detection (Without NLI)

Instead of heavy NLI models, use OpenRouter LLM:

```python
async def detect_conflicts(evidences: list[dict]) -> list[dict]:
    """Use LLM to detect conflicts between evidence chunks."""
    if len(evidences) < 2:
        return []
    
    prompt = f"""Compare these {len(evidences)} evidence snippets.
Identify any contradictions or conflicts between them.

Evidence:
{format_evidences(evidences)}

Return JSON array of conflicts:
[{{"source_a": "id1", "source_b": "id2", "reason": "...", "confidence": 0.8}}]
If no conflicts, return []."""

    response = await llm.generate(prompt, model="gpt-4o-mini")
    return parse_json(response)
```

---

## requirements.txt (Minimal)

```txt
# Core
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
python-multipart>=0.0.6
httpx>=0.26.0
python-dotenv>=1.0.0

# Vector DB
lancedb>=0.4.0

# Document Processing
docling>=0.1.0
Pillow>=10.0.0
easyocr>=1.7.0

# Audio/Video
openai-whisper>=20231117
moviepy>=1.0.3
opencv-python>=4.8.0
yt-dlp>=2024.1.0

# Embeddings
sentence-transformers>=2.2.0
transformers>=4.36.0
torch>=2.1.0
```

---

## Frontend Structure (Vanilla JS)

```
frontend/
├── index.html          # Main page
├── styles.css          # Plain CSS
├── app.js              # Main logic
├── api.js              # API calls
└── viewers/
    ├── pdf-viewer.js   # PDF.js wrapper
    ├── text-viewer.js  # DOCX/TXT display
    ├── image-viewer.js # Image + bbox overlay
    └── media-player.js # Audio/Video with timestamp
```

---

## Folder Structure (Simplified)

```
chakravyuh/
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Settings
│   ├── models.py               # Pydantic schemas
│   ├── db.py                   # LanceDB client
│   ├── ingestion/
│   │   ├── documents.py        # Docling parser
│   │   ├── images.py           # Image + OCR
│   │   ├── audio.py            # Whisper
│   │   └── video.py            # MoviePy
│   ├── retrieval.py            # Vector search
│   ├── reasoning.py            # Conflict + uncertainty
│   ├── generation.py           # OpenRouter LLM
│   └── export.py               # Obsidian export
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── data/                       # Uploaded files
├── obsidian_vault/             # Export destination
├── .env
└── requirements.txt
```
