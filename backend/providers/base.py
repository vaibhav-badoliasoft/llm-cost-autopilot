from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class LLMResponse:
    output_text: str
    model_id: str
    provider: str
    tokens_in: int
    tokens_out: int
    cost: float
    latency_ms: int
    success: bool
    error: Optional[str] = None
    raw_response: Optional[dict] = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def __str__(self) -> str:
        status = "OK" if self.success else f"ERROR: {self.error}"
        return (
            f"[{self.provider}/{self.model_id}] "
            f"{status} | "
            f"in={self.tokens_in} out={self.tokens_out} | "
            f"cost=${self.cost:.6f} | "
            f"latency={self.latency_ms}ms"
        )


class BaseProvider(ABC):

    @abstractmethod
    async def send_request(
        self,
        prompt: str,
        model_id: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        ...

    @abstractmethod
    def health_check(self) -> bool:
        ...

    @staticmethod
    def _timer_ms(start: float) -> int:
        return int((time.time() - start) * 1000)