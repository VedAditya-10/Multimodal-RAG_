


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from loguru import logger
from typing import Optional
from datetime import datetime
import uuid
import sys

from config import DATA_DIR, FRAMES_DIR, OPENROUTER_API_KEY
from models import (
    QueryRequest, QueryResponse, 
    IngestResponse, EvidenceResponse,
    Citation
)
from db import get_db
from llm import get_llm
from embedder import get_embedder

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Create app
app = FastAPI(
    title="CHAKRAVYUH",
    description="Multimodal RAG with Universal Evidence Citing",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files for serving uploaded content
app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")


# === Root ===

@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": "CHAKRAVYUH",
        "version": "1.0.0",
        "description": "Multimodal RAG with Universal Evidence Citing",
        "endpoints": {
            "health": "/health",
            "ingest": "POST /ingest",
            "query": "POST /query",
            "evidence": "GET /evidence/{chunk_id}",
            "export": "POST /export/obsidian",
            "docs": "/docs"
        }
    }


# === Health Check ===

@app.get("/health")
async def health():
    """Health check endpoint."""
    db = get_db()
    
    # Check OpenRouter
    llm_ok = OPENROUTER_API_KEY is not None and len(OPENROUTER_API_KEY) > 10
    
    return {
        "status": "ok",
        "db_rows": db.count(),
        "openrouter_configured": llm_ok,
    }


# === Ingest ===

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Ingest a document, image, audio, or video file."""
    # Import here to avoid circular imports
    from ingestion import ingest_file
    
    # Generate source ID
    source_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    file_ext = Path(file.filename).suffix.lower()
    save_path = DATA_DIR / f"{source_id}{file_ext}"
    
    content = await file.read()
    
    # Check file size for videos
    if file_ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
        from config import MAX_VIDEO_SIZE_MB
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_VIDEO_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"Video file too large: {size_mb:.1f}MB (max {MAX_VIDEO_SIZE_MB}MB)"
            )
    
    with open(save_path, "wb") as f:
        f.write(content)
    
    logger.info(f"Saved file: {save_path}")
    
    # Process file
    try:
        chunks, modalities = await ingest_file(save_path, source_id, file.filename)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        # Cleanup failed file
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
    
    # Store in LanceDB
    db = get_db()
    db.insert(chunks)
    
    return IngestResponse(
        status="success",
        source_id=source_id,
        filename=file.filename,
        chunks_created=len(chunks),
        modalities=list(modalities)
    )


# === Query ===

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the knowledge base with two-layer retrieval."""
    from retrieval import retrieve
    from reasoning import calculate_uncertainty, should_refuse, get_refusal_reason
    
    llm = get_llm()
    
    # Two-layer retrieval:
    # Layer 1: Vector similarity search
    # Layer 2: Modality + confidence re-ranking
    results = retrieve(
        request.query,
        limit=request.max_results,
        modalities=request.modalities,
        rerank=True  # Enable modality-aware re-ranking
    )
    
    if not results:
        return QueryResponse(
            answer="I cannot answer this question. No relevant evidence found in the knowledge base.",
            confidence=0.0,
            breakdown={},
            citations=[],
            conflicts=[],
            refused=True,
            refusal_reason="No relevant evidence found",
            modalities_searched=request.modalities or ["all"]
        )
    
    # Calculate uncertainty
    uncertainty = calculate_uncertainty(results, results)
    
    # Check if we should refuse
    if should_refuse(uncertainty):
        reason = get_refusal_reason(uncertainty)
        return QueryResponse(
            answer=f"I cannot answer this question with sufficient confidence. {reason}",
            confidence=uncertainty["confidence"],
            breakdown=uncertainty["breakdown"],
            citations=[],
            conflicts=[],
            refused=True,
            refusal_reason=reason,
            modalities_searched=list(set(r.get("modality", "unknown") for r in results))
        )
    
    # Build context from evidence
    context = "\n\n".join([
        f"[{i+1}] ({r.get('source_file', 'unknown')}, {r.get('modality', 'unknown')}): {r.get('text_content', '')[:500]}"
        for i, r in enumerate(results)
    ])
    
    # Detect conflicts
    conflicts = await llm.detect_conflicts(results)
    
    # Generate answer
    answer = await llm.generate(request.query, context=context)
    
    # Build citations
    citations = []
    for i, r in enumerate(results):
        location = {}
        if r.get("page_number"):
            location["page"] = r["page_number"]
        if r.get("timestamp_start"):
            location["timestamp_start"] = r["timestamp_start"]
            location["timestamp_end"] = r.get("timestamp_end")
        if r.get("bbox"):
            location["bbox"] = r["bbox"]
        if r.get("line_start"):
            location["line_start"] = r["line_start"]
            location["line_end"] = r.get("line_end")
        
        # Check if this chunk has conflicts
        conflict_ids = []
        for c in conflicts:
            if c.get("source_a") == i + 1:
                conflict_ids.append(results[c.get("source_b", 1) - 1].get("chunk_id", ""))
            elif c.get("source_b") == i + 1:
                conflict_ids.append(results[c.get("source_a", 1) - 1].get("chunk_id", ""))
        
        citations.append(Citation(
            chunk_id=r.get("chunk_id", ""),
            source_id=r.get("source_id", ""),
            source_file=r.get("source_file", "unknown"),
            modality=r.get("modality", "unknown"),
            location=location,
            text_snippet=r.get("text_content", "")[:200],
            confidence=r.get("similarity", 0.5),  # Use pre-computed similarity
            conflicts_with=conflict_ids
        ))
    
    
    return QueryResponse(
        query=request.query,  # Include original query for export
        answer=answer,
        confidence=uncertainty["confidence"],
        breakdown=uncertainty["breakdown"],
        citations=citations,
        conflicts=conflicts,
        refused=False,
        modalities_searched=list(set(r.get("modality", "unknown") for r in results))
    )


# === Evidence ===

@app.get("/evidence/{chunk_id}", response_model=EvidenceResponse)
async def get_evidence(chunk_id: str):
    """Get raw evidence for a chunk."""
    db = get_db()
    chunk = db.get_by_id(chunk_id)
    
    if not chunk:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    # Build location
    location = {}
    if chunk.get("page_number"):
        location["page"] = chunk["page_number"]
    if chunk.get("timestamp_start"):
        location["timestamp_start"] = chunk["timestamp_start"]
        location["timestamp_end"] = chunk.get("timestamp_end")
    if chunk.get("bbox"):
        location["bbox"] = chunk["bbox"]
    
    # Determine content URL
    if chunk.get("image_path"):
        content_url = f"/frames/{Path(chunk['image_path']).name}"
    else:
        # For documents, point to the source file
        content_url = f"/files/{chunk.get('source_id', '')}{Path(chunk.get('source_file', '')).suffix}"
    
    return EvidenceResponse(
        chunk_id=chunk_id,
        source_file=chunk.get("source_file", "unknown"),
        modality=chunk.get("modality", "unknown"),
        content_url=content_url,
        location=location,
        text_content=chunk.get("text_content")
    )


# === Export ===

from fastapi.responses import Response
from models import QueryResponse as QR

@app.post("/export/markdown")
async def export_markdown(query_response: QR):
    """Export conversation as downloadable markdown file."""
    from export import format_conversation_markdown
    
    try:
        markdown = format_conversation_markdown(
            query=query_response.query if hasattr(query_response, 'query') else "Query",
            answer=query_response.answer,
            citations=[c.dict() for c in query_response.citations],
            confidence=query_response.confidence,
            conflicts=query_response.conflicts
        )
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"TRACE_Query_{timestamp}.md"
        
        return Response(
            content=markdown,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        logger.error(f"Markdown export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export/obsidian")
async def export_obsidian(request: dict):
    """Save conversation to Obsidian via REST API (uses request credentials or .env)."""
    from export import save_to_obsidian
    from models import QueryResponse as QR
    
    try:
        # Extract Obsidian credentials from request
        api_key = request.get('obsidian_api_key')
        api_url = request.get('obsidian_api_url')
        
        # Parse QueryResponse from request
        query_response = QR(**request)
        
        result = await save_to_obsidian(
            query=query_response.query if hasattr(query_response, 'query') else "Query",
            answer=query_response.answer,
            citations=[c.dict() for c in query_response.citations],
            confidence=query_response.confidence,
            conflicts=query_response.conflicts,
            api_key=api_key,
            api_url=api_url
        )
        return result
    except Exception as e:
        logger.error(f"Obsidian export failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# === Main ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
