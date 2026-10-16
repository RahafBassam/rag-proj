import numpy as np

from src.config import TOP_K
from src.embeddings import embed


def retrieve(question: str, chunks: list[str], vectors: list[list[float]], top_k: int = TOP_K):
    """Return the top_k most similar chunks as (index, score, text)."""
    q = np.array(embed([question])[0])
    m = np.array(vectors)

    scores = (m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q))
    best = np.argsort(scores)[::-1][:top_k]

    return [(int(i), float(scores[i]), chunks[i]) for i in best]