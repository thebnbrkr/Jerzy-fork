# llm.py - Language model interfaces and implementations
from jerzy.common import *


from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import re

class LLM:
    """Base interface for language models with token tracking."""

    def __init__(self):
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0
        }
        self.token_usage_history = []

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement generate()")

    def generate_with_tools(self, prompt: str, tools: List[Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement generate_with_tools()")

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_usage

    def get_token_usage_history(self) -> List[Dict[str, Any]]:
        return self.token_usage_history

    def reset_token_usage(self) -> None:
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0
        }
        self.token_usage_history = []

class OpenAILLM(LLM):
    """Implementation for OpenAI-compatible APIs with token tracking."""

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Run `pip install openai`.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str) -> str:
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        if hasattr(response, 'usage'):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "timestamp": datetime.now().isoformat()
            }
            if hasattr(response.usage, 'estimated_cost'):
                usage["estimated_cost"] = response.usage.estimated_cost

            self.token_usage["prompt_tokens"] += usage["prompt_tokens"]
            self.token_usage["completion_tokens"] += usage["completion_tokens"]
            self.token_usage["total_tokens"] += usage["total_tokens"]
            if "estimated_cost" in usage:
                self.token_usage["estimated_cost"] += usage["estimated_cost"]

            self.token_usage_history.append(usage)

        return response.choices[0].message.content
