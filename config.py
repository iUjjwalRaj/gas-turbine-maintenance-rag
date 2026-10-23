"""Configuration settings for Gas Turbine Maintenance Assistant RAG application."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Default PDF document path
DEFAULT_PDF_FILENAME = "GER-3620-P.pdf"
DEFAULT_PDF_PATH = DATA_DIR / DEFAULT_PDF_FILENAME
PDF_PATH = Path(os.getenv("PDF_PATH", str(DEFAULT_PDF_PATH)))

# Official download URL for GE Vernova GER-3620 Rev P
OFFICIAL_PDF_URL = (
    "https://www.gevernova.com/content/dam/gepower-new/global/en_US/downloads/"
    "gas-new-site/resources/reference/ger-3620p-heavy-duty-gas-turbine-operating-"
    "and-maintenance-considerations.pdf"
)

# ChromaDB persistence configuration
CHROMA_PERSIST_DIR = BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gas_turbine_maintenance")

# Document chunking parameters (~800 characters with 120 character overlap)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Local embedding model (sentence-transformers)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# Retrieval parameters
TOP_K = int(os.getenv("TOP_K", "4"))
# Chroma cosine distance threshold for relevance filtering (distance > threshold means irrelevant)
# For sentence-transformers cosine distance: 0 = identical, ~0.5-0.9 = semantically related technical passages, > 1.1 = unrelated.
MAX_COSINE_DISTANCE = float(os.getenv("MAX_COSINE_DISTANCE", "1.05"))

# Optional LLM configuration
USE_LLM = os.getenv("USE_LLM", "false").lower() in ("true", "1", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Metadata tracking file for index cache invalidation
INDEX_METADATA_FILE = CHROMA_PERSIST_DIR / "index_meta.json"
