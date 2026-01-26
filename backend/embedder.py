"""Embedding generation for text and images."""
from typing import Optional
import numpy as np
from loguru import logger

from config import TEXT_EMBEDDING_MODEL, IMAGE_EMBEDDING_MODEL


class Embedder:
   
    
    def __init__(self):
        self._text_model = None
        self._clip_model = None
        self._clip_processor = None
    
    @property
    def text_model(self):
        """Lazy load text embedding model."""
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading text model: {TEXT_EMBEDDING_MODEL}")
            self._text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
        return self._text_model
    
    @property
    def clip_model(self):
        """Lazy load CLIP model."""
        if self._clip_model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            logger.info(f"Loading CLIP model: {IMAGE_EMBEDDING_MODEL}")
            self._clip_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
            self._clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
            # Move to GPU if available
            if torch.cuda.is_available():
                self._clip_model = self._clip_model.cuda()
                logger.info("CLIP model moved to GPU")
        return self._clip_model, self._clip_processor
    
    def embed_text(self, text: str) -> list[float]:
        """Generate normalized text embedding for cosine similarity."""
        if not text or not text.strip():
            return []
        embedding = self.text_model.encode(text, convert_to_numpy=True)
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts with normalization."""
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return []
        embeddings = self.text_model.encode(texts, convert_to_numpy=True)
        # L2 normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
        embeddings = embeddings / norms
        return embeddings.tolist()
    
    def embed_image(self, image) -> list[float]:
        """Generate image embedding using CLIP."""
        import torch
        from PIL import Image
        
        model, processor = self.clip_model
        
        # Ensure PIL Image
        if isinstance(image, str):
            image = Image.open(image)
        
        inputs = processor(images=image, return_tensors="pt")
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model.get_image_features(**inputs)
        embedding = outputs.detach().cpu().numpy()[0]
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding.tolist()
    
    def embed_image_from_path(self, path: str) -> list[float]:
        """Generate image embedding from file path."""
        from PIL import Image
        image = Image.open(path)
        return self.embed_image(image)


# Singleton
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Get or create embedder."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
