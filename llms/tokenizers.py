from typing import Any

import tiktoken
from transformers import LlamaTokenizer  # type: ignore


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            # Handle custom models that are not recognized by tiktoken
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # For deepseek-chat and other models not recognized by tiktoken,
                # use cl100k_base (GPT-4 tokenizer) as default
                if "deepseek" in model_name.lower():
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                else:
                    # Fallback to cl100k_base for other unknown models
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
