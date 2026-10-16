import json
from pathlib import Path

from src.config import EMBED_MODEL, INDEX_PATH


def save_index(chunks: list[str], vectors: list[list[float]], path: str = INDEX_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {"embed_model": EMBED_MODEL, "chunks": chunks, "vectors": vectors}
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def load_index(path: str = INDEX_PATH) -> tuple[list[str], list[list[float]]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No index found at {path}. Run: python -m src.build_index")

    data = json.loads(p.read_text(encoding="utf-8"))
    if data["embed_model"] != EMBED_MODEL:
        raise RuntimeError(
            f"Index was built with '{data['embed_model']}' but config uses '{EMBED_MODEL}'. "
            "Rebuild the index: python -m src.build_index"
        )
    return data["chunks"], data["vectors"]
