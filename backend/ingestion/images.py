"""Image parser with OCR and CLIP embeddings."""
from pathlib import Path
from loguru import logger

from config import FRAMES_DIR


async def parse_image(file_path: Path) -> list[dict]:
    """
    Parse image files.
    
    - Generate CLIP embedding
    - Extract text via EasyOCR
    - Handle GIFs by extracting first frame
    """
    from PIL import Image
    
    chunks = []
    ext = file_path.suffix.lower()
    
    # Handle GIFs - extract first frame
    if ext == ".gif":
        image = Image.open(file_path)
        image.seek(0)  # First frame
        # Save as PNG for processing
        png_path = FRAMES_DIR / f"{file_path.stem}_frame0.png"
        image.convert("RGB").save(png_path)
        file_path = png_path
    
    # Load image
    image = Image.open(file_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Get image dimensions for bbox normalization
    width, height = image.size
    
    # Run OCR
    ocr_results = await run_ocr(file_path)
    
    if ocr_results:
        # Create chunk for each OCR region
        for result in ocr_results:
            bbox = result.get("bbox")  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            text = result.get("text", "")
            confidence = result.get("confidence", 0.5)
            
            if not text.strip():
                continue
            
            # Normalize bbox to 0-1
            if bbox:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                normalized_bbox = [
                    min(x_coords) / width,
                    min(y_coords) / height,
                    max(x_coords) / width,
                    max(y_coords) / height,
                ]
            else:
                normalized_bbox = None
            
            chunks.append({
                "text_content": text,
                "modality": "image",
            })
    
    # Use vision LLM to describe the image (for diagrams, charts, photos)
    vision_description = await get_vision_description(str(file_path))
    
    if vision_description:
        # Add vision description as primary chunk
        chunks.append({
            "text_content": vision_description,
            "modality": "image",
        })
        logger.info(f"Vision description added ({len(vision_description)} chars)\")")
    
    logger.info(f"Image parsed: {len(chunks)} chunks ({len(ocr_results)} OCR regions)")
    return chunks


async def get_vision_description(image_path: str) -> str:
    """Get vision LLM description of an image."""
    try:
        from llm import get_llm
        llm = get_llm()
        return await llm.describe_image(image_path)
    except Exception as e:
        logger.warning(f"Vision description failed: {e}")
        return ""


async def run_ocr(file_path: Path) -> list[dict]:
    """Run EasyOCR on image."""
    try:
        import easyocr
        
        # Initialize reader (lazy load)
        reader = easyocr.Reader(['en'], gpu=True)
        
        results = reader.readtext(str(file_path))
        
        ocr_results = []
        for bbox, text, confidence in results:
            ocr_results.append({
                "bbox": bbox,
                "text": text,
                "confidence": confidence,
            })
        
        return ocr_results
        
    except ImportError:
        logger.warning("EasyOCR not installed, skipping OCR")
        return []
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return []
