from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    display_name: str
    cost_per_input_token: float
    cost_per_output_token: float
    context_window: int
    quality_tier: int
    avg_latency_ms: int
    notes: str = ""

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4o",
        cost_per_input_token=5.00 / 1_000_000,
        cost_per_output_token=15.00 / 1_000_000,
        context_window=128_000,
        quality_tier=3,
        avg_latency_ms=2800,
        notes="Gold standard — used as quality judge",
    ),
    "gpt-4o-mini": ModelConfig(
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        cost_per_input_token=0.15 / 1_000_000,
        cost_per_output_token=0.60 / 1_000_000,
        context_window=128_000,
        quality_tier=2,
        avg_latency_ms=900,
        notes="Best mid-tier value",
    ),
    "claude-sonnet": ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        display_name="Claude Sonnet 4.5",
        cost_per_input_token=3.00 / 1_000_000,
        cost_per_output_token=15.00 / 1_000_000,
        context_window=200_000,
        quality_tier=3,
        avg_latency_ms=2500,
        notes="Premium Anthropic tier",
    ),
    "claude-haiku": ModelConfig(
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        cost_per_input_token=0.80 / 1_000_000,
        cost_per_output_token=4.00 / 1_000_000,
        context_window=200_000,
        quality_tier=2,
        avg_latency_ms=700,
        notes="Fastest Anthropic model",
    ),
    "llama3.2": ModelConfig(
        provider="ollama",
        model_id="llama3.2",
        display_name="Llama 3.2 (Local)",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        context_window=131_072,
        quality_tier=1,
        avg_latency_ms=1200,
        notes="Local inference via Ollama — zero cost",
    ),
}


def get_model(key: str) -> ModelConfig:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Model '{key}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]


def get_models_by_tier(tier: int) -> list[ModelConfig]:
    return [m for m in MODEL_REGISTRY.values() if m.quality_tier == tier]


def get_cheapest_in_tier(tier: int) -> Optional[ModelConfig]:
    models = get_models_by_tier(tier)
    if not models:
        return None
    return min(models, key=lambda m: m.cost_per_input_token + m.cost_per_output_token)