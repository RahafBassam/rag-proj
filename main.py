import openai

from src.config import TOP_K
from src.vector_store import load_index
from src.retrieval import retrieve
from src.generate import generate_answer


def main():
    try:
        chunks, vectors = load_index()
    except (FileNotFoundError, RuntimeError) as e:
        raise SystemExit(f"Error: {e}")

    print(f"Loaded {len(chunks)} chunks. Ask a question (type 'exit' to quit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            retrieved = retrieve(question, chunks, vectors, top_k=TOP_K)
            answer = generate_answer(question, retrieved)
        except openai.OpenAIError as e:
            print(f"Bot: API error: {e}\n")
            continue

        print(f"Bot: {answer}")
        used = ", ".join(f"chunk {i} ({score:.2f})" for i, score, _ in retrieved)
        print(f"     [sources: {used}]\n")


if __name__ == "__main__":
    main()