from pathlib import Path

from src.config import DATA_PATH
from src.chunking import chunk_by_lines
from src.embeddings import embed
from src.vector_store import save_index


def main():
    text = Path(DATA_PATH).read_text(encoding="utf-8")
    chunks = chunk_by_lines(text)
    vectors = embed(chunks)
    save_index(chunks, vectors)
    print(f"Saved index with {len(chunks)} chunks ({len(vectors[0])} dims each).")


if __name__ == "__main__":
    main()