"""PDF parser with robust fallback chain: pdfplumber → PyPDF2 → OCR."""
from pathlib import Path
from loguru import logger
import re


async def parse_pdf(file_path: Path) -> list[dict]:
    """
    Extract text from PDF with 3-layer fallback:
    1. pdfplumber (best for structured PDFs)
    2. PyPDF2 (fallback for simple PDFs)
    3. OCR (for scanned documents)
    
    Returns semantic chunks with page numbers.
    """
    # Try pdfplumber first
    text = _extract_with_pdfplumber(file_path)
    if text.strip():
        logger.info(f"Extracted {len(text)} chars with pdfplumber")
        return _chunk_by_pages(text, "text")
    
    # Try PyPDF2 as fallback
    text = _extract_with_pypdf2(file_path)
    if text.strip():
        logger.info(f"Extracted {len(text)} chars with PyPDF2")
        return _chunk_by_pages(text, "text")
    
    # Try OCR as last resort
    text = _extract_with_ocr(file_path)
    if text.strip():
        logger.info(f"Extracted {len(text)} chars with OCR")
        return _chunk_by_pages(text, "text")  # Use "text" not "ocr_text"
    
    logger.warning("No readable text found in PDF")
    return []


def _extract_with_pdfplumber(file_path: Path) -> str:
    """Extract text using pdfplumber (best for tables and layout)."""
    try:
        import pdfplumber
        
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text_parts.append(f"--- Page {page_num} ---\n{text}")
        
        return '\n\n'.join(text_parts)
    except ImportError:
        logger.warning("pdfplumber not installed")
        return ""
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
        return ""


def _extract_with_pypdf2(file_path: Path) -> str:
    """Extract text using PyPDF2 (fallback)."""
    try:
        import PyPDF2
        
        text_parts = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text:
                    text_parts.append(f"--- Page {page_num} ---\n{text}")
        
        return '\n\n'.join(text_parts)
    except ImportError:
        logger.warning("PyPDF2 not installed")
        return ""
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}")
        return ""


def _extract_with_ocr(file_path: Path) -> str:
    """Extract text using OCR (for scanned PDFs)."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        
        logger.info("Attempting OCR extraction for scanned PDF")
        text_parts = []
        
        # Convert first 20 pages to images
        images = convert_from_path(file_path, first_page=1, last_page=20)
        
        for page_num, image in enumerate(images, 1):
            try:
                # Convert to grayscale for better OCR
                if image.mode != 'L':
                    image = image.convert('L')
                
                text = pytesseract.image_to_string(image)
                if text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{text}")
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num}: {e}")
                continue
        
        return '\n\n'.join(text_parts) if text_parts else ""
        
    except ImportError:
        logger.warning("pytesseract or pdf2image not installed")
        return ""
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
        return ""


def _chunk_by_pages(full_text: str, modality: str) -> list[dict]:
    """
    Split text by page markers and create semantic chunks.
    
    Merges paragraphs within each page into ~300-500 char chunks.
    """
    chunks = []
    
    # Split by page markers
    pages = re.split(r'--- Page (\d+) ---', full_text)
    
    current_page = 1
    for i in range(1, len(pages), 2):
        if i + 1 >= len(pages):
            break
        
        page_num = int(pages[i])
        page_text = pages[i + 1].strip()
        
        if not page_text:
            continue
        
        # Clean text
        page_text = _clean_text(page_text)
        
        # Split by paragraphs
        paragraphs = page_text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Merge until 300-500 chars
            if len(current_chunk) + len(para) > 500:
                if current_chunk:
                    chunks.append({
                        "text_content": current_chunk.strip(),
                        "modality": modality,
                        "page_number": page_num,
                    })
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                "text_content": current_chunk.strip(),
                "modality": modality,
                "page_number": page_num,
            })
    
    return chunks


def _clean_text(text: str) -> str:
    """
    Clean extracted text to remove problematic characters.
    
    Removes:
    - Null bytes (\x00)
    - Control characters (except newlines/tabs)
    - Excessive whitespace
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove control characters except newlines and tabs
    cleaned_chars = []
    for char in text:
        if char.isprintable() or char in ['\n', '\t', '\r']:
            cleaned_chars.append(char)
        elif ord(char) < 32:
            cleaned_chars.append(' ')
    
    cleaned_text = ''.join(cleaned_chars)
    
    # Remove excessive whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n\s+\n', '\n\n', cleaned_text)
    
    return cleaned_text.strip()
