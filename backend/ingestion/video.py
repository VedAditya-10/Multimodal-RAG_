"""Video parser using MoviePy for frames and Whisper for audio."""
from pathlib import Path
from loguru import logger
import cv2

from config import (
    FRAMES_DIR,
    MAX_VIDEO_DURATION_SEC,
    VIDEO_FRAME_RATE,
    MAX_KEYFRAMES,
    VIDEO_MAX_WIDTH,
)
from ingestion.audio import parse_audio


async def parse_video(file_path: Path, source_id: str) -> list[dict]:
    """
    Parse video files.
    
    - Extract audio → transcribe with Whisper
    - Extract keyframes at intervals
    - Run OCR on frames
    - Generate CLIP embeddings for frames
    
    Respects configured limits:
    - MAX_VIDEO_DURATION_SEC: 600 (10 min)
    - VIDEO_FRAME_RATE: 0.5 fps (1 frame per 2 sec)
    - MAX_KEYFRAMES: 30
    - VIDEO_MAX_WIDTH: 1280 (720p)
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        logger.error("MoviePy not installed")
        return []
    
    chunks = []
    
    try:
        logger.info(f"Processing video: {file_path}")
        clip = VideoFileClip(str(file_path))
        
        # Cap duration
        duration = min(clip.duration, MAX_VIDEO_DURATION_SEC)
        if clip.duration > MAX_VIDEO_DURATION_SEC:
            logger.warning(f"Video truncated: {clip.duration:.1f}s → {duration:.1f}s")
        
        # 1. Extract and transcribe audio
        audio_chunks = await extract_and_transcribe_audio(clip, file_path, duration)
        chunks.extend(audio_chunks)
        
        # 2. Extract keyframes
        frame_chunks = await extract_keyframes(clip, file_path, source_id, duration)
        chunks.extend(frame_chunks)
        
        clip.close()
        
        logger.info(f"Video processed: {len(chunks)} chunks ({len(audio_chunks)} audio, {len(frame_chunks)} frames)")
        return chunks
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        return []


async def extract_and_transcribe_audio(clip, file_path: Path, duration: float) -> list[dict]:
    """Extract audio track and transcribe."""
    import tempfile
    
    if clip.audio is None:
        logger.info("No audio track in video")
        return []
    
    try:
        # Export audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)
        
        # Only process up to duration limit
        audio_clip = clip.audio.subclip(0, duration)
        audio_clip.write_audiofile(
            str(temp_path),
            fps=16000,
            verbose=False,
            logger=None
        )
        
        # Transcribe
        chunks = await parse_audio(temp_path)
        
        # Update modality
        for chunk in chunks:
            chunk["modality"] = "audio_transcript"
        
        # Clean up
        temp_path.unlink(missing_ok=True)
        
        return chunks
        
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return []


async def extract_keyframes(clip, file_path: Path, source_id: str, duration: float) -> list[dict]:
    """Extract keyframes at regular intervals."""
    from PIL import Image
    import numpy as np
    
    chunks = []
    
    # Calculate frame times
    interval = 1 / VIDEO_FRAME_RATE  # e.g., 2 seconds
    frame_times = []
    t = 0
    while t < duration and len(frame_times) < MAX_KEYFRAMES:
        frame_times.append(t)
        t += interval
    
    logger.info(f"Extracting {len(frame_times)} frames")
    
    for i, t in enumerate(frame_times):
        try:
            # Get frame at time t
            frame = clip.get_frame(t)  # Returns numpy array (H, W, C)
            
            # Resize if needed
            h, w = frame.shape[:2]
            if w > VIDEO_MAX_WIDTH:
                scale = VIDEO_MAX_WIDTH / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Convert to PIL for saving
            pil_image = Image.fromarray(frame.astype(np.uint8))
            
            # Save frame
            frame_filename = f"{source_id}_frame_{i:03d}.jpg"
            frame_path = FRAMES_DIR / frame_filename
            pil_image.save(frame_path, quality=85)
            
            # Run OCR on frame
            ocr_text = await run_frame_ocr(frame_path)
            
            chunk = {
                "image_path": str(frame_path),
                "modality": "video_frame",
                "timestamp_start": t,
                "timestamp_end": t + interval,
            }
            
            if ocr_text:
                chunk["text_content"] = ocr_text
            
            chunks.append(chunk)
            
        except Exception as e:
            logger.warning(f"Failed to extract frame at {t}s: {e}")
            continue
    
    return chunks


async def run_frame_ocr(frame_path: Path) -> str | None:
    """Run OCR on a video frame."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)
        results = reader.readtext(str(frame_path))
        
        if results:
            # Combine all detected text
            texts = [r[1] for r in results if r[2] > 0.5]  # Confidence > 0.5
            return " ".join(texts) if texts else None
        return None
        
    except Exception as e:
        logger.warning(f"Frame OCR failed: {e}")
        return None
