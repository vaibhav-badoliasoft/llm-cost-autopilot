import os
import time
from typing import Optional

import httpx

from providers.base import BaseProvider, LLMResponse


class OllamaProvider(BaseProvider):
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")

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

        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            latency = self._timer_ms(start)
            tokens_in = data.get("prompt_eval_count", 0)
            tokens_out = data.get("eval_count", 0)
            output_text = data.get("message", {}).get("content", "")

            return LLMResponse(
                output_text=output_text,
                model_id=model_id,
                provider="ollama",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=0.0,
                latency_ms=latency,
                success=True,
                raw_response=data,
            )

        except Exception as e:
            return LLMResponse(
                output_text="",
                model_id=model_id,
                provider="ollama",
                tokens_in=0,
                tokens_out=0,
                cost=0.0,
                latency_ms=self._timer_ms(start),
                success=False,
                error=str(e),
            )

    def health_check(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False