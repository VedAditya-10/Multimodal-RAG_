"""Uncertainty calculation and hallucination guard."""
import numpy as np
from typing import Optional
from loguru import logger

from config import REFUSAL_THRESHOLD, WARNING_THRESHOLD


def calc_embedding_score(search_results: list[dict]) -> float:
    """
    Convert LanceDB search results to similarity scores.
    
    Since embeddings are now L2-normalized, we use the similarity
    field directly (already computed as 1 - distance).
    """
    if not search_results:
        return 0.0
    
    similarities = []
    for r in search_results:
        # Use pre-computed similarity (from db.search)
        similarity = r.get("similarity", 0.5)
        similarity = max(0.0, min(1.0, similarity))
        similarities.append(similarity)
    
    # Weight top results more heavily (positional decay)
    weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(similarities)]
    weighted_avg = sum(s * w for s, w in zip(similarities, weights)) / sum(weights[:len(similarities)])
    
    return round(weighted_avg, 3)


def calc_extraction_confidence(chunks: list[dict]) -> float:
    """
    Average confidence from OCR (EasyOCR) and ASR (Whisper).
    
    OCR: EasyOCR provides confidence directly (0-1).
    ASR: Whisper provides avg_logprob per segment.
         Convert logprob to probability: exp(avg_logprob)
         Typical range: -0.5 (good) to -1.5 (poor)
    """
    confidences = []
    
    for chunk in chunks:
        # OCR confidence (EasyOCR)
        if chunk.get("ocr_confidence") is not None:
            confidences.append(chunk["ocr_confidence"])
        
        # ASR confidence (Whisper) - prefer avg_logprob
        if chunk.get("avg_logprob") is not None:
            logprob = chunk["avg_logprob"]
            # Normalize: clamp to [-1.5, 0], then scale to [0, 1]
            normalized = max(0.0, min(1.0, (logprob + 1.5) / 1.5))
            confidences.append(normalized)
        elif chunk.get("asr_confidence") is not None:
            confidences.append(chunk["asr_confidence"])
        elif chunk.get("no_speech_prob") is not None:
            # Fallback proxy
            confidences.append(1.0 - chunk["no_speech_prob"])
    
    if not confidences:
        return 1.0  # No OCR/ASR involved, assume text is clean
    
    return round(sum(confidences) / len(confidences), 3)


def calc_modality_agreement(chunks: list[dict]) -> float:
    """
    Measure agreement across modalities using embedding similarity.
    Deterministic, fast, no LLM.
    """
    modalities = set(c.get("modality", "unknown") for c in chunks)
    
    # Base score from modality diversity
    if len(modalities) >= 3:
        diversity_bonus = 0.2
    elif len(modalities) == 2:
        diversity_bonus = 0.1
    else:
        diversity_bonus = 0.0
    
    # Collect embeddings (unified text embeddings only)
    embeddings = []
    for c in chunks:
        if c.get("text_embedding"):
            embeddings.append(np.array(c["text_embedding"]))
    
    if len(embeddings) < 2:
        return 0.5 + diversity_bonus
    
    # Pairwise cosine similarity
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            a, b = embeddings[i], embeddings[j]
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            similarities.append(max(0.0, cos_sim))  # Clamp negative
    
    avg_similarity = sum(similarities) / len(similarities)
    scaled = max(0.0, min(1.0, avg_similarity))
    final = min(1.0, scaled * 0.8 + diversity_bonus)
    
    return round(final, 3)


def calc_source_score(chunks: list[dict]) -> float:
    """More sources = higher confidence (with diminishing returns)."""
    unique_sources = len(set(c.get("source_file", "") for c in chunks))
    
    if unique_sources >= 5:
        return 1.0
    elif unique_sources == 4:
        return 0.9
    elif unique_sources == 3:
        return 0.8
    elif unique_sources == 2:
        return 0.6
    elif unique_sources == 1:
        return 0.4
    else:
        return 0.0


# Weight constants
WEIGHTS = {
    "embedding": 0.35,
    "extraction": 0.20,
    "modality": 0.25,
    "sources": 0.20,
}


def calculate_uncertainty(chunks: list[dict], search_results: list[dict]) -> dict:
    """
    Calculate confidence score. Fully deterministic, no LLM calls.
    """
    scores = {
        "embedding": calc_embedding_score(search_results),
        "extraction": calc_extraction_confidence(chunks),
        "modality": calc_modality_agreement(chunks),
        "sources": calc_source_score(chunks),
    }
    
    # Weighted average
    final_score = sum(scores[k] * WEIGHTS[k] for k in scores)
    
    return {
        "confidence": round(final_score, 2),
        "breakdown": scores,
        "should_refuse": final_score < REFUSAL_THRESHOLD,
        "show_warning": final_score < WARNING_THRESHOLD,
        "evidence_count": len(chunks),
        "modalities_found": list(set(c.get("modality", "unknown") for c in chunks)),
    }


def should_refuse(result: dict) -> bool:
    """Check if we should refuse to answer."""
    return result.get("should_refuse", False)


def get_refusal_reason(result: dict) -> str:
    """Get human-readable refusal reason."""
    if not result.get("should_refuse"):
        return ""
    
    reasons = []
    breakdown = result.get("breakdown", {})
    
    if breakdown.get("embedding", 1) < 0.3:
        reasons.append("No relevant evidence found in knowledge base")
    if breakdown.get("sources", 1) < 0.4:
        reasons.append("Insufficient number of sources")
    if breakdown.get("extraction", 1) < 0.5:
        reasons.append("Low confidence in text extraction (OCR/ASR)")
    if breakdown.get("modality", 1) < 0.3:
        reasons.append("Evidence chunks are semantically inconsistent")
    
    return "; ".join(reasons) if reasons else "Evidence quality below threshold"
