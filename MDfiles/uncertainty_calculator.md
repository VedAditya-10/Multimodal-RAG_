# Uncertainty Calculator - Specification (Fixed)

## Overview

The uncertainty calculator produces a **confidence score (0.0 - 1.0)** for each answer based on concrete, measurable factors.

---

## Input Factors

| Factor | Source | Weight |
|--------|--------|--------|
| **Embedding Similarity** | LanceDB search score | 0.35 |
| **OCR/ASR Confidence** | EasyOCR / Whisper | 0.20 |
| **Modality Agreement** | Cross-modal embedding similarity | 0.25 |
| **Source Count** | Number of supporting sources | 0.20 |

---

## Factor Calculations

### 1. Embedding Similarity (35%)

**LanceDB uses cosine distance by default.** Cosine distance = 1 - cosine_similarity.

```python
def calc_embedding_score(search_results: list[dict]) -> float:
    """
    Convert LanceDB cosine distances to similarity scores.
    
    LanceDB stores '_distance' as cosine distance (1 - cosine_similarity).
    Cosine distance range: [0, 2] where 0 = identical, 2 = opposite.
    We convert to similarity: [0, 1] where 1 = identical.
    """
    if not search_results:
        return 0.0
    
    similarities = []
    for r in search_results:
        distance = r["_distance"]  # Cosine distance: 0 to 2
        similarity = 1 - (distance / 2)  # Convert to 0-1 range
        similarity = max(0.0, min(1.0, similarity))  # Clamp
        similarities.append(similarity)
    
    # Weight top results more heavily (positional decay)
    weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(similarities)]
    weighted_avg = sum(s * w for s, w in zip(similarities, weights)) / sum(weights[:len(similarities)])
    
    return round(weighted_avg, 3)
```

**Alternative: Store similarity explicitly during ingestion**
```python
# When storing in LanceDB, also store the raw similarity
chunk["stored_similarity"] = 1 - (distance / 2)
```

**Interpretation:**
- 0.8+ → Very similar, high confidence
- 0.5-0.8 → Moderately similar
- <0.5 → Weak match, low confidence

---

### 2. OCR/ASR Confidence (20%)

#### OCR (EasyOCR)
EasyOCR returns confidence per detected text box (0.0 - 1.0). Use directly.

#### ASR (Whisper)
**Preferred:** Use `avg_logprob` (average log probability of tokens).

```python
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
        
        # ASR confidence (Whisper)
        if chunk.get("avg_logprob") is not None:
            # Convert log probability to 0-1 range
            # avg_logprob typically ranges from -1.5 (poor) to 0 (perfect)
            logprob = chunk["avg_logprob"]
            # Normalize: clamp to [-1.5, 0], then scale to [0, 1]
            normalized = max(0.0, min(1.0, (logprob + 1.5) / 1.5))
            confidences.append(normalized)
        
        # Fallback: use no_speech_prob as proxy (DOCUMENTED AS WEAKER)
        elif chunk.get("no_speech_prob") is not None:
            # Note: This is a weaker proxy. Prefer avg_logprob when available.
            confidences.append(1.0 - chunk["no_speech_prob"])
    
    if not confidences:
        return 1.0  # No OCR/ASR involved, assume text is clean
    
    return round(sum(confidences) / len(confidences), 3)
```

**Whisper segment output includes:**
```python
segment = {
    "text": "...",
    "start": 0.0,
    "end": 2.5,
    "avg_logprob": -0.32,      # USE THIS (preferred)
    "no_speech_prob": 0.02,    # Fallback only
    "compression_ratio": 1.2,
}
```

---

### 3. Modality Agreement (25%)

**NO LLM CALLS.** Use embedding similarity between evidence chunks.

```python
import numpy as np

def calc_modality_agreement(chunks: list[dict]) -> float:
    """
    Measure agreement across modalities using embedding similarity.
    
    Computes pairwise cosine similarity between chunk embeddings,
    then averages. Higher = more agreement.
    
    This is deterministic, fast, and does not use LLM.
    """
    modalities = set(c["modality"] for c in chunks)
    
    # Base score from modality diversity
    if len(modalities) >= 3:
        diversity_bonus = 0.2  # Text + Image + Audio/Video
    elif len(modalities) == 2:
        diversity_bonus = 0.1
    else:
        diversity_bonus = 0.0
    
    # Compute embedding similarity between chunks
    embeddings = []
    for c in chunks:
        if c.get("text_embedding") is not None:
            embeddings.append(np.array(c["text_embedding"]))
        elif c.get("image_embedding") is not None:
            # Note: CLIP embeddings are in same space as text
            embeddings.append(np.array(c["image_embedding"]))
    
    if len(embeddings) < 2:
        # Can't compute similarity with < 2 embeddings
        return 0.5 + diversity_bonus
    
    # Pairwise cosine similarity
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            a, b = embeddings[i], embeddings[j]
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            similarities.append(cos_sim)
    
    avg_similarity = sum(similarities) / len(similarities)
    
    # Scale to 0-1 (cosine similarity is already -1 to 1)
    # Typical RAG results should be positive, so scale from [0, 1]
    scaled = max(0.0, min(1.0, avg_similarity))
    
    # Combine with diversity bonus
    final = min(1.0, scaled * 0.8 + diversity_bonus)
    
    return round(final, 3)
```

**Why embedding similarity works:**
- CLIP puts text and images in the same vector space
- High similarity = chunks talk about the same thing
- Deterministic: same inputs → same output
- Fast: just numpy operations

---

### 4. Source Count (20%)

```python
def calc_source_score(chunks: list[dict]) -> float:
    """More sources = higher confidence (with diminishing returns)."""
    unique_sources = len(set(c["source_file"] for c in chunks))
    
    # Diminishing returns curve
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
```

---

## Final Score Calculation

```python
class UncertaintyCalculator:
    WEIGHTS = {
        "embedding": 0.35,
        "extraction": 0.20,
        "modality": 0.25,
        "sources": 0.20,
    }
    
    REFUSAL_THRESHOLD = 0.4  # Below this, refuse to answer
    WARNING_THRESHOLD = 0.6  # Below this, show warning
    
    def calculate(self, chunks: list[dict], search_results: list[dict]) -> dict:
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
        final_score = sum(scores[k] * self.WEIGHTS[k] for k in scores)
        
        return {
            "confidence": round(final_score, 2),
            "breakdown": scores,
            "should_refuse": final_score < self.REFUSAL_THRESHOLD,
            "show_warning": final_score < self.WARNING_THRESHOLD,
            "evidence_count": len(chunks),
            "modalities_found": list(set(c["modality"] for c in chunks)),
        }
```

---

## Refusal Logic

```python
def get_refusal_reason(result: dict) -> str | None:
    if not result["should_refuse"]:
        return None
    
    reasons = []
    breakdown = result["breakdown"]
    
    if breakdown["embedding"] < 0.3:
        reasons.append("No relevant evidence found in knowledge base")
    if breakdown["sources"] < 0.4:
        reasons.append("Insufficient number of sources")
    if breakdown["extraction"] < 0.5:
        reasons.append("Low confidence in text extraction (OCR/ASR)")
    if breakdown["modality"] < 0.3:
        reasons.append("Evidence chunks are semantically inconsistent")
    
    return "; ".join(reasons) if reasons else "Evidence quality below threshold"
```

---

## Example Output

```json
{
  "confidence": 0.72,
  "breakdown": {
    "embedding": 0.85,
    "extraction": 0.78,
    "modality": 0.60,
    "sources": 0.50
  },
  "should_refuse": false,
  "show_warning": false,
  "evidence_count": 3,
  "modalities_found": ["pdf", "video"]
}
```

---

## Key Properties

| Property | Value |
|----------|-------|
| **Deterministic** | ✅ Yes - same inputs → same output |
| **LLM-free** | ✅ Yes - no LLM calls in uncertainty calc |
| **Fast** | ✅ Yes - only numpy operations |
| **Judge-safe** | ✅ Yes - explainable formulas |
