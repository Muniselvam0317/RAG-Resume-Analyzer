"""Groq chat completion wrapper (streaming)."""
from typing import Dict, Iterator, List

SYSTEM_PROMPT = """You are an expert technical recruiter assistant analysing resumes.

RULES
1. Answer ONLY from the RESUME EXCERPTS and the CANDIDATE ROSTER provided. Never invent
   skills, employers, dates, degrees or contact details.
2. Always name the candidate (and file) each fact comes from.
3. If the excerpts don't contain the answer, say "I couldn't find that in the uploaded
   resumes." Do not guess.
4. When asked to rank or compare candidates, base it only on evidence in the excerpts and
   briefly justify. Mention if the evidence may be incomplete (only top excerpts are shown).
5. Be concise. Use bullet points or a table when listing several candidates."""


class GroqLLM:
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        if not api_key or api_key.startswith("PASTE_YOUR"):
            raise RuntimeError(
                "Groq API key missing. Open config.py and set GROQ_API_KEY "
                "(or export the GROQ_API_KEY environment variable)."
            )
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
