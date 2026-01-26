"""Configuration settings for CHAKRAVYUH."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
FRAMES_DIR = Path(os.getenv("FRAMES_DIR", "./frames"))
OBSIDIAN_VAULT = Path(os.getenv("OBSIDIAN_VAULT", "./obsidian_vault"))

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LLM Models
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "anthropic/claude-3.5-sonnet")
FAST_MODEL = os.getenv("FAST_MODEL", "openai/gpt-4o-mini")
LLM_MODELS = [PRIMARY_MODEL, FAST_MODEL, "google/gemini-flash-1.5"]

# Video limits
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", 100))
MAX_VIDEO_DURATION_SEC = int(os.getenv("MAX_VIDEO_DURATION_SEC", 600))
VIDEO_FRAME_RATE = 0.5  # 1 frame per 2 seconds
MAX_KEYFRAMES = 30
VIDEO_MAX_WIDTH = 1280

# Embedding models
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEXT_EMBEDDING_DIM = 384
IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"
IMAGE_EMBEDDING_DIM = 512

# Uncertainty thresholds
REFUSAL_THRESHOLD = 0.4
WARNING_THRESHOLD = 0.6

# LanceDB
LANCEDB_PATH = BASE_DIR / "lancedb"
