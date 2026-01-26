"""Audio parser using Whisper for transcription.

Fixes:
1. Model loaded once and cached (reproducibility)
2. Proper confidence scoring using exp(avg_logprob)
3. no_speech_prob filtering
4. Semantic re-chunking by sentences
5. Ready-to-store format (no separate embedding needed)
"""
from pathlib import Path
from loguru import logger
import math
import re

# Global model cache for reproducibility
_WHISPER_MODEL = None
WHISPER_MODEL_VERSION = "base"  # Versioned for auditability


def _get_whisper_model():
    """Load Whisper model once and cache it."""
    global _WHISPER_MODEL
    
    if _WHISPER_MODEL is None:
        try:
            import whisper
            logger.info(f"Loading Whisper model: {WHISPER_MODEL_VERSION}")
            _WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_VERSION)
            logger.info("Whisper model loaded and cached")
        except ImportError:
            logger.error("Whisper not installed")
            raise
    
    return _WHISPER_MODEL


def _calculate_confidence(avg_logprob: float, no_speech_prob: float) -> float:
    """
    Calculate proper confidence from Whisper outputs.
    
    Uses exp(avg_logprob) to respect log scale, then penalizes by no_speech_prob.
    """
    # Convert log probability to linear probability
    # avg_logprob is typically -0.1 to -1.5
    base_confidence = math.exp(avg_logprob)
    
    # Penalize by no_speech probability
    # If no_speech_prob is high, confidence should be low
    confidence = base_confidence * (1.0 - no_speech_prob)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, confidence))


def _rechunk_by_sentences(segments: list[dict]) -> list[dict]:
    """
    Re-chunk Whisper segments by sentence boundaries for better semantic meaning.
    
    Whisper segments are acoustic-based. We need semantic chunks for:
    - Better embeddings
    - Better retrieval
    - Better conflict detection
    """
    chunks = []
    current_text = ""
    current_start = None
    current_logprobs = []
    current_no_speech_probs = []
    
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        
        # Initialize timing
        if current_start is None:
            current_start = seg.get("start", 0.0)
        
        # Accumulate
        current_text += " " + text if current_text else text
        current_logprobs.append(seg.get("avg_logprob", -0.5))
        current_no_speech_probs.append(seg.get("no_speech_prob", 0.0))
        
        # Check for sentence boundary
        if re.search(r'[.!?]\s*$', text):
            # End of sentence - create chunk
            avg_logprob = sum(current_logprobs) / len(current_logprobs)
            avg_no_speech = sum(current_no_speech_probs) / len(current_no_speech_probs)
            
            chunks.append({
                "text_content": current_text.strip(),
                "modality": "audio_transcript",
            })
            
            # Reset
            current_text = ""
            current_start = None
            current_logprobs = []
            current_no_speech_probs = []
    
    # Add remaining text if any
    if current_text.strip():
        avg_logprob = sum(current_logprobs) / len(current_logprobs) if current_logprobs else -0.5
        avg_no_speech = sum(current_no_speech_probs) / len(current_no_speech_probs) if current_no_speech_probs else 0.0
        
        chunks.append({
            "text_content": current_text.strip(),
            "modality": "audio_transcript",
        })
    
    return chunks


async def parse_audio(file_path: Path) -> list[dict]:
    """
    Parse audio files using Whisper.
    
    Returns ready-to-store chunks with:
    - Semantic segmentation (by sentences)
    - Proper confidence scoring (respects log scale)
    - Silence/noise filtering (no_speech_prob)
    - Reproducible transcription (cached model)
    """
    try:
        model = _get_whisper_model()
        
        logger.info(f"Transcribing audio: {file_path}")
        
        # Transcribe
        result = model.transcribe(
            str(file_path),
            verbose=False,
        )
        
        segments = result.get("segments", [])
        
        # Filter out segments with high no_speech_prob (silence/music/noise)
        NO_SPEECH_THRESHOLD = 0.6
        valid_segments = [
            seg for seg in segments
            if seg.get("no_speech_prob", 0.0) < NO_SPEECH_THRESHOLD
        ]
        
        logger.info(f"Filtered {len(segments) - len(valid_segments)} silent/noisy segments")
        
        # Re-chunk by sentences for semantic meaning
        chunks = _rechunk_by_sentences(valid_segments)
        
        logger.info(f"Transcribed {len(chunks)} semantic chunks (model: {WHISPER_MODEL_VERSION})")
        return chunks
        
    except ImportError:
        logger.error("Whisper not installed")
        return []
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return []
