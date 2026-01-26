"""Pydantic models for API requests/responses and LanceDB schema."""
from typing import Optional
from pydantic import BaseModel


# === LanceDB Schema ===

class EvidenceChunk(BaseModel):
    """Schema for evidence stored in LanceDB."""
    # Identity
    chunk_id: str
    source_id: str
    source_file: str
    
    # Modality
    source_type: str      # pdf, docx, txt, markdown, image, scan, audio, video
    modality: str         # text, image, audio_transcript, video_frame, ocr
    
    # Content
    text_content: Optional[str] = None
    image_path: Optional[str] = None
    
    # Location (unified for all modalities)
    page_number: Optional[int] = None           # PDF
    section: Optional[str] = None               # DOCX heading
    line_start: Optional[int] = None            # TXT, MD
    line_end: Optional[int] = None
    timestamp_start: Optional[float] = None     # Audio, Video (seconds)
    timestamp_end: Optional[float] = None
    bbox: Optional[list[float]] = None          # [x1, y1, x2, y2] normalized 0-1
    
    # Confidence from extraction
    ocr_confidence: Optional[float] = None      # EasyOCR confidence
    asr_confidence: Optional[float] = None      # Whisper avg_logprob converted
    
    # Will be set after embedding
    text_embedding: Optional[list[float]] = None
    image_embedding: Optional[list[float]] = None


# === API Request/Response Models ===

class IngestRequest(BaseModel):
    """Request for file ingestion (multipart form, not JSON)."""
    pass  # File comes via UploadFile


class IngestResponse(BaseModel):
    """Response after ingestion."""
    status: str
    source_id: str
    filename: str
    chunks_created: int
    modalities: list[str]


class QueryRequest(BaseModel):
    """Query request."""
    query: str
    modalities: Optional[list[str]] = None  # Filter by modality
    max_results: int = 5


class Citation(BaseModel):
    """Single citation in response."""
    chunk_id: str
    source_id: str
    source_file: str
    modality: str
    location: dict  # page, timestamp, bbox, etc.
    text_snippet: str
    confidence: float
    conflicts_with: list[str] = []


class QueryResponse(BaseModel):
    """Query response with answer and citations."""
    query: Optional[str] = None  # Original query (for export)
    answer: str
    confidence: float
    breakdown: dict  # Uncertainty breakdown
    citations: list[Citation]
    conflicts: list[dict]
    refused: bool = False
    refusal_reason: Optional[str] = None
    modalities_searched: list[str]


class EvidenceResponse(BaseModel):
    """Response for fetching raw evidence."""
    chunk_id: str
    source_file: str
    modality: str
    content_url: str  # URL to fetch the actual file
    location: dict
    text_content: Optional[str] = None


class ConflictResponse(BaseModel):
    """Conflict detection response."""
    conflicts: list[dict]
    total: int


class ExportResponse(BaseModel):
    """Obsidian export response."""
    status: str
    notes_created: int
    vault_path: str
