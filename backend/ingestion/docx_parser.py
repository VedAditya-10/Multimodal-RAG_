"""DOCX parser using python-docx."""
from pathlib import Path
from loguru import logger


async def parse_docx(file_path: Path) -> list[dict]:
    """
    Extract text from DOCX files.
    
    Returns chunks by paragraphs with section headers preserved.
    """
    try:
        from docx import Document
        
        doc = Document(file_path)
        chunks = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            # Detect headings
            is_heading = para.style.name.startswith('Heading')
            
            chunks.append({
                "text_content": text,
                "modality": "text",
            })
        
        logger.info(f"Extracted {len(chunks)} paragraphs from DOCX")
        return chunks
        
    except ImportError:
        logger.error("python-docx not installed")
        return []
    except Exception as e:
        logger.error(f"DOCX parsing failed: {e}")
        return []
