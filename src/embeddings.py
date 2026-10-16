from openai import OpenAI

from src.config import OPENAI_API_KEY, EMBED_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)


def embed(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Turn a list of texts into a list of vectors, in batches."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = _client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors