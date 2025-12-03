import argparse
from typing import Any

try:
    from vertexai.preview.generative_models import Image
    from llms import generate_from_gemini_completion
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion')

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)

APIInput = str | list[Any] | dict[str, Any]


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
) -> str:
    response: str
    if lm_config.provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            # Safely get configuration parameters with defaults
            temperature = lm_config.gen_config.get("temperature", 1.0)
            top_p = lm_config.gen_config.get("top_p", 0.9)
            context_length = lm_config.gen_config.get("context_length", 0)
            max_tokens = lm_config.gen_config.get("max_tokens", 384)
            stop_token = lm_config.gen_config.get("stop_token", None)

            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=temperature,
                top_p=top_p,
                context_length=context_length,
                max_tokens=max_tokens,
                stop_token=stop_token,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            # Safely get configuration parameters with defaults
            temperature = lm_config.gen_config.get("temperature", 1.0)
            max_tokens = lm_config.gen_config.get("max_tokens", 384)
            top_p = lm_config.gen_config.get("top_p", 0.9)
            stop_token = lm_config.gen_config.get("stop_token", None)

            response = generate_from_openai_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop_token=stop_token,
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        # Safely get configuration parameters with defaults
        model_endpoint = lm_config.gen_config.get("model_endpoint", "")
        temperature = lm_config.gen_config.get("temperature", 1.0)
        top_p = lm_config.gen_config.get("top_p", 0.9)
        stop_sequences = lm_config.gen_config.get("stop_sequences", [])
        max_new_tokens = lm_config.gen_config.get("max_new_tokens", 384)

        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=model_endpoint,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            max_new_tokens=max_new_tokens,
        )
    elif lm_config.provider == "google":
        assert isinstance(prompt, list)
        assert all(
            [isinstance(p, str) or isinstance(p, Image) for p in prompt]
        )
        # Safely get configuration parameters with defaults
        temperature = lm_config.gen_config.get("temperature", 1.0)
        max_tokens = lm_config.gen_config.get("max_tokens", 384)
        top_p = lm_config.gen_config.get("top_p", 0.9)

        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    return response
