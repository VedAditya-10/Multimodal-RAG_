# CHAKRAVYUH - Task Checklist (Simplified)

## Phase 1: Setup (Day 1)
- [ ] Create folder structure
- [ ] Setup Python venv + requirements.txt
- [ ] Install FFmpeg
- [ ] Configure .env with OpenRouter key
- [ ] FastAPI app with /health
- [ ] LanceDB schema + connection
- [ ] Test OpenRouter API

## Phase 2: Ingestion (Days 2-3)
- [ ] Document parser (Docling: PDF, DOCX, MD)
- [ ] Image parser (Pillow + EasyOCR + CLIP)
- [ ] Audio parser (Whisper segments)
- [ ] Video parser (MoviePy frames + Whisper)
- [ ] `/ingest` endpoint
- [ ] Test with sample files

## Phase 3: RAG Engine (Days 4-5)
- [ ] Embedder (text + image)
- [ ] Retriever (LanceDB search)
- [ ] Conflict detection (via LLM)
- [ ] Uncertainty calculator
- [ ] Hallucination guard (refusal logic)
- [ ] LLM router (OpenRouter)
- [ ] Citation builder
- [ ] `/query` endpoint

## Phase 4: Frontend (Day 6)
- [ ] HTML layout (chat + evidence panel)
- [ ] Plain CSS styling
- [ ] Vanilla JS API calls
- [ ] PDF viewer (PDF.js CDN)
- [ ] Text viewer (HTML highlight)
- [ ] Image viewer (img + canvas bbox)
- [ ] Audio/Video player (native tags + timestamp)
- [ ] Citation click → open viewer

## Phase 5: Export & Polish (Day 7)
- [ ] Obsidian export (query + evidence notes)
- [ ] `/export/obsidian` endpoint
- [ ] Error handling
- [ ] Demo dataset
- [ ] README
- [ ] Test adversarial inputs
