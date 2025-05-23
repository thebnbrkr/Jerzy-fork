# core.py - Fundamental data structures for Jerzt
from jerzy.common import *



from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import json
import time

class Prompt:
    """Minimal prompt template system."""
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        """Format the prompt template with provided values."""
        return self.template.format(**kwargs)


class ToolCache:
    """Cache for storing and retrieving tool call results."""

    def __init__(self, max_size: int = 100, ttl: Optional[int] = None):
        self.cache: Dict[str, Tuple[Any, float]] = {}  # {cache_key: (result, timestamp)}
        self.max_size = max_size
        self.ttl = ttl

    def _generate_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        sorted_args = json.dumps(args, sort_keys=True)
        return f"{tool_name}:{sorted_args}"

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._generate_key(tool_name, args)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if self.ttl is not None and time.time() - timestamp > self.ttl:
                del self.cache[key]
                return None
            return result
        return None

    def set(self, tool_name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
        key = self._generate_key(tool_name, args)
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = (result, time.time())

    def clear(self) -> None:
        self.cache.clear()

    def remove(self, tool_name: str, args: Dict[str, Any]) -> None:
        key = self._generate_key(tool_name, args)
        if key in self.cache:
            del self.cache[key]


class State:
    """Manages the evolving state of knowledge during agent execution."""

    def __init__(self):
        self.data = {}
        self.history = []
        self.version = 0

    def set(self, key: str, value: Any) -> None:
        if "." in key:
            parts = key.split(".")
            current = self.data
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.data[key] = value

        self.history.append({
            "action": "set",
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "version": self.version
        })
        self.version += 1

    def get(self, key: str, default: Any = None) -> Any:
        if "." in key:
            parts = key.split(".")
            current = self.data
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
            return current
        return self.data.get(key, default)

    def has_key(self, key: str) -> bool:
        if "." in key:
            parts = key.split(".")
            current = self.data
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]
            return True
        return key in self.data

    def append_to(self, key: str, value: Any) -> None:
        current = self.get(key, [])
        if not isinstance(current, list):
            current = [current]
        current.append(value)
        self.set(key, current)

    def to_dict(self) -> Dict[str, Any]:
        import copy
        return copy.deepcopy(self.data)



class Tool:
    """Represents a callable tool for the LLM to use."""

    def __init__(self, name: str, func: Callable, description: str, cacheable: bool = True,
                 allow_repeated_calls: bool = False):
        self.name = name
        self.func = func
        self.description = description
        self.signature = self._get_signature()
        self.cacheable = cacheable
        self.allow_repeated_calls = allow_repeated_calls  # Flag to allow intentional repeats

    def _get_signature(self) -> Dict[str, Any]:
        """Extract parameter information from the function signature."""
        sig = inspect.signature(self.func)
        params = {}

        for param_name, param in sig.parameters.items():
            # Skip self for methods
            if param_name == 'self':
                continue

            param_type = param.annotation if param.annotation != inspect.Parameter.empty else "string"
            if hasattr(param_type, "__name__"):
                type_name = param_type.__name__
            else:
                type_name = str(param_type).replace("<class '", "").replace("'>", "")

            params[param_name] = {
                "type": type_name,
                "required": param.default == inspect.Parameter.empty
            }

        return params

    def __call__(self, *args, **kwargs):
        """Execute the tool with the provided arguments and handle caching."""
        # Extract cache from kwargs if present
        cache = kwargs.pop('cache', None)

        # Check if we should try to use cache
        if cache is not None and self.cacheable:
            # Try to get cached result
            cached_result = cache.get(self.name, kwargs)
            if cached_result is not None:
                # Add a flag to indicate this was a cache hit
                cached_result["cached"] = True
                return cached_result

        # No cache hit, execute the tool
        try:
            result = {
                "status": "success",
                "result": self.func(*args, **kwargs),
                "timestamp": datetime.now().isoformat(),
                "cached": False
            }

            # Store in cache if applicable
            if cache is not None and self.cacheable:
                cache.set(self.name, kwargs, result)

            return result
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat(),
                "cached": False
            }

            # We don't cache errors
            return error_result

