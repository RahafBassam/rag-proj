from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_TEMPLATE = """Answer the question using ONLY the context below.
Do not add any information that is not explicitly in the context.
If the answer is not in the context, say you don't know.

<context>
{context}
</context>"""


def generate_answer(question: str, retrieved: list[tuple[int, float, str]]) -> str:
    context = "\n\n---\n\n".join(text for _, _, text in retrieved)

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content