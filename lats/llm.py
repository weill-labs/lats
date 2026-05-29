"""LLM provider abstraction.

`LLM.chat(messages, n=...)` returns a list of `n` completion strings — this `n`
is what lets one expansion sample several children with a single API round-trip
(OpenAI's native `n` parameter). `MockLLM` implements the same contract with a
caller-supplied handler so the whole MCTS loop can be tested without any network
or API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from tenacity import retry, stop_after_attempt, wait_random_exponential


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLM:
    #: number of chat() calls made (round-trips) and completions returned.
    num_calls: int = 0
    num_completions: int = 0

    def chat(
        self,
        messages: List[Message],
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> List[str]:
        raise NotImplementedError

    def _record(self, n: int) -> None:
        self.num_calls += 1
        self.num_completions += n


class OpenAIChat(LLM):
    """OpenAI Chat Completions via the modern (>=1.0) SDK.

    The official LATS code targeted the removed `openai.ChatCompletion.create`
    API; this is the equivalent on the current client. The key is read from
    OPENAI_API_KEY in the environment.
    """

    def __init__(self, model: str):
        from openai import OpenAI  # imported lazily so tests need no key

        self.model = model
        self.client = OpenAI()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def chat(
        self,
        messages: List[Message],
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> List[str]:
        self._record(n)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
        )
        return [c.message.content or "" for c in resp.choices]


class MockLLM(LLM):
    """Deterministic stand-in for tests.

    `handler(messages, n)` receives the prompt and the requested completion count
    and returns a list of `n` strings. Every call is recorded for assertions.
    """

    def __init__(self, handler: Callable[[List[Message], int], List[str]]):
        self.handler = handler
        self.calls: List[List[Message]] = []

    def chat(
        self,
        messages: List[Message],
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> List[str]:
        self._record(n)
        self.calls.append(messages)
        out = self.handler(messages, n)
        assert len(out) == n, (
            f"MockLLM handler returned {len(out)} completions, expected {n}"
        )
        return out
