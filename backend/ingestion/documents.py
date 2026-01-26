"""Document parser dispatcher - routes to specialized parsers."""
from pathlib import Path
from loguru import logger


async def parse_document(file_path: Path) -> list[dict]:
    """
    Route to appropriate parser based on file extension.
    
    Supports: PDF, DOCX, MD, TXT
    """
    ext = file_path.suffix.lower()
    
    if ext == '.pdf':
        from ingestion.pdf import parse_pdf
        return await parse_pdf(file_path)
    
    elif ext == '.docx':
        from ingestion.docx_parser import parse_docx
        return await parse_docx(file_path)
    
    elif ext in {'.md', '.markdown'}:
        from ingestion.markdown import parse_markdown
        return await parse_markdown(file_path)
    
    elif ext == '.txt':
        from ingestion.markdown import parse_plain_text
        return await parse_plain_text(file_path)
    
    else:
        logger.error(f"Unsupported file type: {ext}")
        return []
 