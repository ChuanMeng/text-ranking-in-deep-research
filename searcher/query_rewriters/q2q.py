from __future__ import annotations

from dataclasses import dataclass
import logging

import openai

Q2Q_TEMPLATE = """
You are given a web search query.
Rewrite it into a single MS MARCO-style natural language question.

Guidelines:
- Preserve ALL numbers, quantities, and units exactly as they appear in the web search query.
- If a number appears in the web search query, it MUST appear verbatim in the output.
- Do NOT explain or elaborate.
- Output only ONE question.

Below are EXAMPLES of valid MS MARCO-style queries (outputs only).
Do NOT copy them.

Examples:
- do goldfish grow
- what is wifi vs bluetooth
- why did the us volunterilay enter ww1
- does legionella pneumophila cause pneumonia
- who is robert gray
- how long is the life cycle of a flea
- how much does interior concrete flooring cost
- difference between a mcdouble and a double cheeseburger

Web search query: {query}
MS MARCO-style query:
""".strip()

Q2Q_CONTEXT_TEMPLATE = """
You are given an agent's reasoning context and a web search query issued by the agent.
Rewrite the web search query into a single MS MARCO-style natural language question.

Guidelines:
- Preserve the original search intent in the agent's reasoning context exactly.
- Preserve ALL numbers, quantities, and units exactly as they appear in the web search query.
- If a number appears in the web search query, it MUST appear verbatim in the output.
- Do NOT explain or elaborate.
- Output only ONE question.

Below are EXAMPLES of valid MS MARCO-style queries (outputs only).
Do NOT copy them.

Examples:
- do goldfish grow
- what is wifi vs bluetooth
- why did the us volunterilay enter ww1
- does legionella pneumophila cause pneumonia
- who is robert gray
- how long is the life cycle of a flea
- how much does interior concrete flooring cost
- difference between a mcdouble and a double cheeseburger

Agent reasoning: {reasoning}
Web search query: {query}
MS MARCO-style query:
""".strip()


@dataclass
class QueryToQuestion:
    """Rewrite a query into a MS MARCO-style natural language question."""

    model_name: str = "openai/gpt-oss-20b"
    model_url: str = "http://localhost:8000/v1"
    max_new_tokens: int = 64

    def __post_init__(self) -> None:
        self.client = openai.OpenAI(
            base_url=self.model_url,
            api_key="EMPTY",
        )

    def __call__(self, query: str) -> str:

        cleaned = query.strip()

        if not query:
            logging.warning("Query is empty")
            return ""

        prompt = Q2Q_TEMPLATE.format(query=cleaned)

        def _run_once() -> str:
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=[{"role": "user", "content": prompt}],
                    max_output_tokens=self.max_new_tokens,
                    reasoning={"effort": "low", "summary": "detailed"},
                )
            except Exception as exc:
                logging.warning("Request failed: %s", exc)
                return ""

            response_dict = response.model_dump(mode="python")
            text = ""
            for item in response_dict["output"]:
                if isinstance(item, dict) and item["type"] == "message":
                    parts = item["content"]
                    text_chunks = [
                        str(part["text"])
                        for part in parts
                        if isinstance(part, dict) and part["type"] == "output_text"
                    ]
                    text = " ".join(chunk for chunk in text_chunks if chunk).strip()
                    if text:
                        break

            return " ".join(text.splitlines()).strip()

        text = _run_once()
        if not text:
            logging.warning("rewrite empty; retrying once")
            text = _run_once()

        if not text:
            logging.warning("rewrite returned empty; fallback to raw query")
            return cleaned
    

        return text

    def rewrite(self, query: str) -> str:
        """Explicit method for rewriting queries."""
        return self(query)

    def rewrite_with_context(self, query: str, reasoning_text: str) -> str:
        cleaned = query.strip()
        reasoning = reasoning_text.strip()

        if not cleaned:
            logging.warning("Query is empty")
            return ""

        assert reasoning is not None

        prompt = Q2Q_CONTEXT_TEMPLATE.format(
            query=cleaned,
            reasoning=" ".join(reasoning.split()),
        )

        def _run_once() -> str:
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=[{"role": "user", "content": prompt}],
                    max_output_tokens=self.max_new_tokens,
                    reasoning={"effort": "low", "summary": "detailed"},
                )
            except Exception as exc:
                logging.warning("Request failed: %s", exc)
                return ""

            response_dict = response.model_dump(mode="python")

            text = ""
            for item in response_dict["output"]:
                if isinstance(item, dict) and item["type"] == "message":
                    parts = item["content"] or []
                    text_chunks = [
                        str(part["text"])
                        for part in parts
                        if isinstance(part, dict) and part["type"] == "output_text"
                    ]
                    text = " ".join(chunk for chunk in text_chunks if chunk).strip()
                    if text:
                        break

            return " ".join(text.splitlines()).strip()

        text = _run_once()

        if not text:
            logging.warning("rewrite with context empty; retrying once")
            text = _run_once()

        if not text:
            logging.warning("rewrite with context empty; fallback to raw query")
            return cleaned

        return text

