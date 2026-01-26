"""LLM client using OpenRouter API.

Supports multi-model fallback and conflict detection.
"""
import httpx
import base64
import asyncio
from typing import Optional
from loguru import logger

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PRIMARY_MODEL, FAST_MODEL, LLM_MODELS


SYSTEM_PROMPT = """You are a precise research assistant that answers questions based ONLY on the provided evidence.

Rules:
1. Base your answer ONLY on the provided context
2. Cite sources using [1], [2], etc. matching the evidence order
3. If evidence is insufficient, say "I cannot answer this based on the available data"
4. If sources conflict, acknowledge the conflict and present both views
5. Never make up information not in the context
6. Be concise but complete"""


class LLMClient:
    """OpenRouter API client with multi-model fallback."""
    
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
        }
    
    async def generate(
        self,
        prompt: str,
        context: str = "",
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> str:
        """Generate response with automatic fallback."""
        models_to_try = [model] if model else LLM_MODELS
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        
        if context:
            messages.append({
                "role": "user", 
                "content": f"Evidence:\n{context}\n\n---\n\nQuestion: {prompt}"
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient(timeout=120) as client:
            for attempt_model in models_to_try:
                retries = 3
                for attempt in range(retries):
                    try:
                        logger.info(f"Trying model: {attempt_model}")
                        response = await client.post(
                            f"{self.base_url}/chat/completions",
                            headers=self.headers,
                            json={
                                "model": attempt_model,
                                "messages": messages,
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 429:  # Rate limited
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.warning(f"Rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        logger.warning(f"Model {attempt_model} failed: {e}")
                        break
                    except Exception as e:
                        logger.warning(f"Model {attempt_model} failed: {e}")
                        break
            
            raise Exception("All LLM models failed")
    
    async def detect_conflicts(self, evidences: list[dict]) -> list[dict]:
        """Use LLM to detect conflicts between evidence chunks."""
        if len(evidences) < 2:
            return []
        
        evidence_text = "\n\n".join([
            f"[{i+1}] ({e.get('source_file', 'unknown')}, {e.get('modality', 'unknown')}): {e.get('text_content', '')[:300]}"
            for i, e in enumerate(evidences)
        ])
        
        prompt = f"""Analyze these {len(evidences)} evidence snippets for contradictions.

{evidence_text}

If there are contradictions, return a JSON array like:
[{{"source_a": 1, "source_b": 2, "claim": "revenue growth", "reason": "Source 1 says 23% growth, Source 2 says costs are concerning"}}]

If no contradictions, return: []

Return ONLY the JSON array, no other text."""

        try:
            response = await self.generate(prompt, model=FAST_MODEL, temperature=0.1)
            # Parse JSON from response
            import json
            # Find JSON array in response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
            return []
        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            return []
    
    async def describe_image(self, image_path: str) -> str:
        """Use vision LLM to describe an image/diagram."""
        import base64
        from pathlib import Path
        
        # Read and encode image
        image_bytes = Path(image_path).read_bytes()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Determine mime type
        ext = Path(image_path).suffix.lower()
        mime_types = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.webp': 'image/webp'}
        mime_type = mime_types.get(ext, 'image/png')
        
        prompt = """Describe this image in detail for a RAG (retrieval-augmented generation) system.

If this is a diagram, flowchart, or technical illustration:
- Describe the overall purpose/topic
- List the main components and their relationships
- Explain the flow or connections shown by arrows
- Mention any text labels visible

If this is a photo or regular image:
- Describe what is shown
- Note any text, numbers, or labels visible
- Describe key visual elements

Be thorough but concise. Your description will be used for semantic search."""

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]}
        ]
        
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                logger.info(f"Describing image with vision LLM: {PRIMARY_MODEL}")
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": PRIMARY_MODEL,
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": 500,
                    }
                )
                response.raise_for_status()
                result = response.json()
                description = result["choices"][0]["message"]["content"]
                logger.info(f"Image description generated ({len(description)} chars)")
                return description
            except Exception as e:
                logger.error(f"Image description failed: {e}")
                return ""
    
    async def health_check(self) -> bool:
        """Check if API is accessible."""
        try:
            response = await self.generate("Say 'ok'", model=FAST_MODEL, max_tokens=10)
            return len(response) > 0
        except:
            return False


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    """Get or create LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
