from pathlib import Path

from src.config import CHUNK_SIZE


def chunk_by_lines(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text at line boundaries and prepend the title to every chunk.

    Assumes the first non-empty line of the file is a title.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title, lines = lines[0], lines[1:]

    chunks, current = [], ""
    for line in lines:
        if current and len(current) + len(line) + 1 > chunk_size:
            chunks.append(f"{title}\n{current}")
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(f"{title}\n{current}")
    return chunks


if __name__ == "__main__":
    text = Path("data/nour_cafe.txt").read_text(encoding="utf-8")
    for i, c in enumerate(chunk_by_lines(text)):
        print(f"--- chunk {i} ({len(c)} chars) ---")
        print(c, "\n")