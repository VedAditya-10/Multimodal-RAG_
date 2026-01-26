"""Markdown and plain text parsers."""
from pathlib import Path
from loguru import logger
import re


async def parse_markdown(file_path: Path) -> list[dict]:
    """
    Parse markdown files by sections.
    
    Splits on headers and preserves structure.
    """
    try:
        text = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Split by headers
        sections = re.split(r'^(#+\s+.+)$', text, flags=re.MULTILINE)
        
        chunks = []
        for part in sections:
            part = part.strip()
            if not part:
                continue
            
            chunks.append({
                "text_content": part,
                "modality": "text",
            })
        
        logger.info(f"Extracted {len(chunks)} sections from Markdown")
        return chunks
        
    except Exception as e:
        logger.error(f"Markdown parsing failed: {e}")
        return []


async def parse_plain_text(file_path: Path) -> list[dict]:
    """
    Parse plain text files by paragraphs.
    """
    try:
        text = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Split by paragraphs
        paragraphs = text.split('\n\n')
        
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if para:
                chunks.append({
                    "text_content": para,
                    "modality": "text",
                })
        
        logger.info(f"Extracted {len(chunks)} paragraphs from plain text")
        return chunks
        
    except Exception as e:
        logger.error(f"Plain text parsing failed: {e}")
        return []
