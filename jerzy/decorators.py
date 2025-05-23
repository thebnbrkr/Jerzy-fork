# decorators.py - Decorators for robustness, fallback, and logging
from jerzy.common import *


import json
import time
import logging
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_fixed

def robust_tool(retries: int = 3, wait_seconds: float = 1.0):
    def decorator(func):
        @retry(stop=stop_after_attempt(retries), wait=wait_fixed(wait_seconds))
        @wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped
    return decorator

def with_fallback(fallback_func):
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.warning(f"Primary tool failed: {e}, using fallback...")
                return fallback_func(*args, **kwargs)
        return wrapped
    return decorator


# ========== Logging Setup ==========
import logging

from functools import wraps

LOG_PATH = "/content/teeno_logs.jsonl"  # Change path if needed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log_tool_call(tool_name):
    """Decorator to log start/end time, args, result, and duration."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                status = "success"
            except Exception as e:
                result = {"error": str(e)}
                status = "error"
            duration = time.time() - start

            log_entry = {
                "tool": tool_name,
                "args": kwargs,
                "result": result,
                "status": status,
                "duration_sec": round(duration, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            logging.info(f"[{tool_name}] status={status} duration={duration:.2f}s")
            return result
        return wrapper
    return decorator
# ===================================

