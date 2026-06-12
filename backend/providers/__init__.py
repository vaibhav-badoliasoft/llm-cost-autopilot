from models.registry import MODEL_REGISTRY, ModelConfig
from providers.base import BaseProvider, LLMResponse
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.ollama_provider import OllamaProvider


_provider_cache: dict[str, BaseProvider] = {}


def get_provider(provider_name: str) -> BaseProvider:
    if provider_name not in _provider_cache:
        if provider_name == "openai":
            _provider_cache["openai"] = OpenAIProvider()
        elif provider_name == "anthropic":
            _provider_cache["anthropic"] = AnthropicProvider()
        elif provider_name == "ollama":
            _provider_cache["ollama"] = OllamaProvider()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    return _provider_cache[provider_name]


async def send_request(
    prompt: str,
    model_key: str,
    system_prompt: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> LLMResponse:
    config: ModelConfig = MODEL_REGISTRY[model_key]
    provider = get_provider(config.provider)
    return await provider.send_request(
        prompt=prompt,
        model_id=config.model_id,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )