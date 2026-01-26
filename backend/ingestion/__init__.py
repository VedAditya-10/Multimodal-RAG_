"""Ingestion router and coordinator."""
from pathlib import Path
from typing import Tuple, Set
from loguru import logger
import uuid

from embedder import get_embedder


# Supported file extensions
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".md", ".markdown", ".txt", ".html"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


async def ingest_file(
    file_path: Path,
    source_id: str,
    original_filename: str
) -> Tuple[list[dict], Set[str]]:
    """
    Ingest a file and return chunks with embeddings.
    
    Args:
        file_path: Path to saved file
        source_id: Unique source identifier
        original_filename: Original uploaded filename
    
    Returns:
        Tuple of (list of chunks, set of modalities)
    """
    ext = file_path.suffix.lower()
    logger.info(f"Ingesting {original_filename} (ext: {ext})")
    
    raw_chunks = []
    modalities = set()
    
    # Route to appropriate parser
    if ext in DOCUMENT_EXTENSIONS:
        from ingestion.documents import parse_document
        raw_chunks = await parse_document(file_path)
        modalities.add("text")
        
    elif ext in IMAGE_EXTENSIONS:
        from ingestion.images import parse_image
        raw_chunks = await parse_image(file_path)
        modalities.add("image")
        if any(c.get("ocr_text") for c in raw_chunks):
            modalities.add("ocr")
            
    elif ext in AUDIO_EXTENSIONS:
        from ingestion.audio import parse_audio
        raw_chunks = await parse_audio(file_path)
        modalities.add("audio_transcript")
        
    elif ext in VIDEO_EXTENSIONS:
        from ingestion.video import parse_video
        raw_chunks = await parse_video(file_path, source_id)
        modalities.update(["video_frame", "audio_transcript"])
        
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Add metadata and embeddings
    embedder = get_embedder()
    final_chunks = []
    
    for i, chunk in enumerate(raw_chunks):
        chunk_id = f"{source_id}_{i:04d}"
        
        # Build final chunk
        final_chunk = {
            "chunk_id": chunk_id,
            "source_id": source_id,
            "source_file": original_filename,
            "source_type": get_source_type(ext),
            "modality": chunk.get("modality", "text"),
            "text_content": chunk.get("text_content"),
            "image_path": chunk.get("image_path"),
            "page_number": chunk.get("page_number"),
            "section": chunk.get("section"),
            "line_start": chunk.get("line_start"),
            "line_end": chunk.get("line_end"),
            "timestamp_start": chunk.get("timestamp_start"),
            "timestamp_end": chunk.get("timestamp_end"),
            "bbox": chunk.get("bbox"),
            "ocr_confidence": chunk.get("ocr_confidence"),
            "asr_confidence": chunk.get("asr_confidence"),
            "avg_logprob": chunk.get("avg_logprob"),
        }
        
        # Generate text embedding for unified search
        # All modalities get text embeddings (from text, OCR, vision description, transcripts)
        if chunk.get("text_content"):
            final_chunk["text_embedding"] = embedder.embed_text(chunk["text_content"])
            final_chunks.append(final_chunk)
        else:
            # Skip chunks without text content to avoid empty embedding issues
            logger.warning(f"Chunk {chunk_id} has no text content, skipping")
    
    logger.info(f"Created {len(final_chunks)} chunks with modalities: {modalities}")
    return final_chunks, modalities


def get_source_type(ext: str) -> str:
    """Get source type from extension."""
    if ext in {".pdf"}:
        return "pdf"
    elif ext in {".docx", ".doc"}:
        return "docx"
    elif ext in {".pptx", ".ppt"}:
        return "pptx"
    elif ext in {".md", ".markdown"}:
        return "markdown"
    elif ext in {".txt"}:
        return "txt"
    elif ext in {".html"}:
        return "html"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in AUDIO_EXTENSIONS:
        return "audio"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    else:
        return "unknown"
