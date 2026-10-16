import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env file.")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-6.1-sol")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

DATA_PATH = str(BASE_DIR / "data" / "nour_cafe.txt")
INDEX_PATH = str(BASE_DIR / "data" / "index.json")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 120))
TOP_K = 2