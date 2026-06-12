import os
import time
from typing import Optional

import anthropic

from providers.base import BaseProvider, LLMResponse
from models.registry import MODEL_REGISTRY


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def send_request(
        self,
        prompt: str,
        model_id: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        config = next(
            (m for m in MODEL_REGISTRY.values() if m.model_id == model_id),
            None,
        )

        kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        start = time.time()
        try:
            response = await self.client.messages.create(**kwargs)

            latency = self._timer_ms(start)
            tokens_in = response.usage.input_tokens
            tokens_out = response.usage.output_tokens
            cost = config.estimate_cost(tokens_in, tokens_out) if config else 0.0
            output_text = response.content[0].text if response.content else ""

            return LLMResponse(
                output_text=output_text,
                model_id=model_id,
                provider="anthropic",
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
                provider="anthropic",
                tokens_in=0,
                tokens_out=0,
                cost=0.0,
                latency_ms=self._timer_ms(start),
                success=False,
                error=str(e),
            )

    def health_check(self) -> bool:
        return bool(self.api_key)