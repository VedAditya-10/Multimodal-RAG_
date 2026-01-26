"""LanceDB client for vector storage and retrieval.

Uses FixedSizeList for proper vector search with cosine metric.
"""
import lancedb
import pyarrow as pa
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

from config import LANCEDB_PATH, TEXT_EMBEDDING_DIM


class LanceDBClient:
    """Client for LanceDB operations with unified cross-modal search."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or LANCEDB_PATH
        self.db = None
        self.table = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database and table."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        
        # Check if table exists
        if "evidence" in self.db.table_names():
            self.table = self.db.open_table("evidence")
            logger.info(f"Opened existing table with {self.table.count_rows()} rows")
        else:
            logger.info("Table 'evidence' does not exist yet, will create on first insert")
    
    def insert(self, chunks: list[dict]) -> int:
        """Insert evidence chunks into the database."""
        if not chunks:
            return 0
        
        # Ensure all embeddings are numpy arrays of correct dimension
        sanitized = []
        for chunk in chunks:
            embedding = chunk.get("text_embedding", [])
            
            # Skip chunks without valid embeddings
            if not embedding or len(embedding) != TEXT_EMBEDDING_DIM:
                logger.warning(f"Skipping chunk with invalid embedding length: {len(embedding) if embedding else 0}")
                continue
            
            # Convert to numpy array for proper FixedSizeList inference
            embedding_array = np.array(embedding, dtype=np.float32)
            
            row = {
                "chunk_id": chunk.get("chunk_id", ""),
                "source_id": chunk.get("source_id", ""),
                "source_file": chunk.get("source_file", ""),
                "text_content": chunk.get("text_content") or "",
                "modality": chunk.get("modality", "unknown"),
                "page_number": chunk.get("page_number"),
                "timestamp_start": chunk.get("timestamp_start"),
                "timestamp_end": chunk.get("timestamp_end"),
                "line_start": chunk.get("line_start"),
                "line_end": chunk.get("line_end"),
                "bbox": chunk.get("bbox"),
                "image_path": chunk.get("image_path"),
                "ocr_confidence": chunk.get("ocr_confidence"),
                "asr_confidence": chunk.get("asr_confidence"),
                "avg_logprob": chunk.get("avg_logprob"),
                "text_embedding": embedding_array,  # numpy array for FixedSizeList
            }
            sanitized.append(row)
        
        if not sanitized:
            logger.warning("No valid chunks to insert after filtering")
            return 0
        
        if self.table is None:
            # Create table - LanceDB will infer FixedSizeList from numpy arrays
            self.table = self.db.create_table(
                "evidence", 
                sanitized,
                mode="overwrite"
            )
            # Verify the schema is correct
            schema = self.table.schema
            embedding_field = schema.field("text_embedding")
            logger.info(f"Created table, {len(sanitized)} rows, embedding type: {embedding_field.type}")
        else:
            self.table.add(sanitized)
            logger.info(f"Added {len(sanitized)} rows")
        
        return len(sanitized)
    
    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        modalities: Optional[list[str]] = None,
        source_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> list[dict]:
        """
        Unified cross-modal search using text embeddings.
        
        Uses cosine similarity on FixedSizeList vector column.
        """
        if self.table is None:
            return []
        
        # Ensure query is numpy array
        query_array = np.array(query_embedding, dtype=np.float32)
        
        # Start search with cosine metric
        query = self.table.search(query_array, vector_column_name="text_embedding")
        
        # Build filter conditions
        filters = []
        
        if modalities:
            mod_conditions = [f"modality = '{m}'" for m in modalities]
            filters.append(f"({' OR '.join(mod_conditions)})")
        
        if source_id:
            filters.append(f"source_id = '{source_id}'")
        
        if min_confidence is not None:
            # Filter by confidence: use OCR if present, else ASR if present, else allow text-only
            filters.append(f"((ocr_confidence IS NOT NULL AND ocr_confidence >= {min_confidence}) OR (ocr_confidence IS NULL AND asr_confidence IS NOT NULL AND asr_confidence >= {min_confidence}) OR (ocr_confidence IS NULL AND asr_confidence IS NULL))")
        
        if filters:
            filter_str = " AND ".join(filters)
            query = query.where(filter_str)
        
        # Execute and return results with distance → similarity conversion
        results = query.limit(limit).to_list()
        
        # Add similarity score (1 - distance for cosine on normalized vectors)
        for r in results:
            distance = r.get("_distance", 0)
            # For normalized vectors, cosine distance is in [0, 2]
            # 0 = identical, 2 = opposite
            r["similarity"] = max(0, 1 - (distance / 2))
        
        return results
    
    def get_by_id(self, chunk_id: str) -> Optional[dict]:
        """Get a specific chunk by ID using direct filter."""
        if self.table is None:
            return None
        
        try:
            results = self.table.search().where(f"chunk_id = '{chunk_id}'").limit(1).to_list()
            return results[0] if results else None
        except Exception as e:
            logger.error(f"get_by_id failed: {e}")
            return None
    
    def get_by_source(self, source_id: str) -> list[dict]:
        """Get all chunks from a source."""
        if self.table is None:
            return []
        
        try:
            return self.table.search().where(f"source_id = '{source_id}'").to_list()
        except Exception as e:
            logger.error(f"get_by_source failed: {e}")
            return []
    
    def delete_source(self, source_id: str) -> int:
        """Delete all chunks from a source. Returns actual count deleted."""
        if self.table is None:
            return 0
        
        try:
            # Count before delete
            before_count = self.table.count_rows()
            
            # Delete
            self.table.delete(f"source_id = '{source_id}'")
            
            # Count after delete
            after_count = self.table.count_rows()
            
            deleted = before_count - after_count
            logger.info(f"Deleted {deleted} rows for source {source_id}")
            return deleted
        except Exception as e:
            logger.error(f"delete_source failed: {e}")
            return 0
    
    def count(self) -> int:
        """Get total row count."""
        if self.table is None:
            return 0
        return self.table.count_rows()
    
    def get_modality_counts(self) -> dict[str, int]:
        """Get count of chunks by modality (sampled for performance)."""
        if self.table is None:
            return {}
        
        try:
            total = self.table.count_rows()
            if total == 0:
                return {}
            
            # For small tables, count all; for large tables, sample
            if total <= 1000:
                rows = self.table.search().limit(total).to_list()
            else:
                # Sample up to 1000 rows for estimate
                rows = self.table.search().limit(1000).to_list()
                logger.info(f"Sampling 1000/{total} rows for modality counts")
            
            counts = {}
            for row in rows:
                mod = row.get("modality", "unknown")
                counts[mod] = counts.get(mod, 0) + 1
            
            # Scale up if sampling
            if total > 1000:
                scale = total / 1000
                counts = {k: int(v * scale) for k, v in counts.items()}
            
            return counts
        except Exception as e:
            logger.error(f"get_modality_counts failed: {e}")
            return {}


# Singleton instance
_db_client: Optional[LanceDBClient] = None


def get_db() -> LanceDBClient:
    """Get or create database client."""
    global _db_client
    if _db_client is None:
        _db_client = LanceDBClient()
    return _db_client
