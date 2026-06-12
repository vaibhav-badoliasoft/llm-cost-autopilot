import os
import time
from typing import Optional

from openai import AsyncOpenAI

from providers.base import BaseProvider, LLMResponse
from models.registry import MODEL_REGISTRY


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def send_request(
        self,
        prompt: str,
        model_id: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        config = next(
            (m for m in MODEL_REGISTRY.values() if m.model_id == model_id),
            None,
        )

        start = time.time()
        try:
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            latency = self._timer_ms(start)
            tokens_in = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens
            cost = config.estimate_cost(tokens_in, tokens_out) if config else 0.0

            return LLMResponse(
                output_text=response.choices[0].message.content,
                model_id=model_id,
                provider="openai",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
                latency_ms=latency,
                success=True,
                raw_response=response.model_dump(),
            )

        except Exception as e:
            return LLMResponse(
                output_text="",
                model_id=model_id,
                provider="openai",
                tokens_in=0,
                tokens_out=0,
                cost=0.0,
                latency_ms=self._timer_ms(start),
                success=False,
                error=str(e),
            )

    def health_check(self) -> bool:
        return bool(self.api_key)