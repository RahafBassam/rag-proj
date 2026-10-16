from src.config import TOP_K
from src.vector_store import load_index
from src.retrieval import retrieve

# (question, text that must appear in the right chunk; None = not in the document)
EVAL_SET = [
    ("How do I return my coffee?", "Refund policy"),
    ("Can I return an opened bag?", "Opened bags"),
    ("Who runs the shop?", "Manager"),
    ("What is your support email?", "Support email"),
    ("When are you open?", "Opening hours"),
    ("When do loyalty points expire?", "Points expire"),
    ("Do you sell pizza?", None),
    ("What is the capital of France?", None),
]


def main():
    chunks, vectors = load_index()
    total = hit_first = hit_any = 0

    for question, expected in EVAL_SET:
        results = retrieve(question, chunks, vectors, top_k=TOP_K)

        if expected is None:
            print(f"[off-topic] {question:<38} top score {results[0][1]:.2f}")
            continue

        total += 1
        found = [expected in text for _, _, text in results]
        hit_first += found[0]
        hit_any += any(found)
        mark = "OK  " if found[0] else ("2ND " if any(found) else "MISS")
        print(f"[{mark}] {question:<38} top score {results[0][1]:.2f}")

    print(f"\nRight chunk ranked 1st:  {hit_first}/{total}")
    print(f"Right chunk in top {TOP_K}:    {hit_any}/{total}")


if __name__ == "__main__":
    main()