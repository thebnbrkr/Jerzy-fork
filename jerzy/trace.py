# trace.py - Execution tracing for reasoning and tool usage

import json
from typing import List, Dict, Any
from .memory import Memory

class Trace:
    """Captures and formats the execution trace for better explainability."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def get_full_trace(self) -> List[Dict[str, Any]]:
        return self.memory.history

    def get_reasoning_trace(self) -> List[str]:
        return self.memory.get_reasoning_chain()

    def get_tool_trace(self) -> List[Dict[str, Any]]:
        return [entry for entry in self.memory.history
                if entry.get("type") == "tool_call" or
                (entry.get("role") == "system" and "Tool result:" in entry.get("content", ""))]

    def format_trace(self, format_type: str = "text") -> str:
        if format_type == "text":
            return self._format_text_trace()
        elif format_type == "markdown":
            return self._format_markdown_trace()
        elif format_type == "json":
            return json.dumps(self.get_full_trace(), indent=2)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def _format_text_trace(self) -> str:
        lines = []
        for entry in self.memory.history:
            role = entry.get("role", "").upper()
            content = entry.get("content", "")
            entry_type = entry.get("type", "")

            if entry_type == "reasoning":
                lines.append(f"🧠 REASONING: {content}")
            elif entry_type == "tool_call":
                lines.append(f"🛠️ TOOL CALL ({entry.get('tool', 'unknown')}): {content}")
            elif "Tool result:" in content:
                cached = " (CACHED)" if entry.get("cached", False) else ""
                lines.append(f"📊 RESULT{cached}: {content.replace('Tool result:', '').strip()}")
            else:
                lines.append(f"{role}: {content}")
            lines.append("-" * 50)
        return "\n".join(lines)

    def _format_markdown_trace(self) -> str:
        lines = ["# Execution Trace", ""]
        current_step = 1

        for entry in self.memory.history:
            role = entry.get("role", "").upper()
            content = entry.get("content", "")
            entry_type = entry.get("type", "")

            if entry_type == "reasoning":
                lines.append(f"## Step {current_step}: Reasoning")
                lines.append(f"_{content}_")
                current_step += 1
            elif entry_type == "tool_call":
                tool = entry.get("tool", "unknown")
                lines.append(f"## Step {current_step}: Tool Call - {tool}")
                lines.append(f"**Parameters:** {entry.get('args', {})}")
                current_step += 1
            elif "Tool result:" in content:
                cached = " (CACHED)" if entry.get("cached", False) else ""
                lines.append(f"### Result{cached}")
                lines.append(f"```\n{content.replace('Tool result:', '').strip()}\n```")
            elif role == "USER":
                lines.append(f"## Query")
                lines.append(f"> {content}")
            elif role == "ASSISTANT" and "Used tool:" not in content:
                lines.append(f"## Final Answer")
                lines.append(content)
            lines.append("")

        return "\n".join(lines)

class AuditTrail:
    """Tracks detailed metrics for token usage, prompts, responses, and tool calls."""

    def __init__(self, storage_path: Optional[str] = None):
        self.entries = []
        self.current_session_id = self._generate_session_id()
        self.storage_path = storage_path

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())

    def start_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new audit session with optional metadata."""
        self.current_session_id = self._generate_session_id()

        session_entry = {
            "type": "session_start",
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.entries.append(session_entry)
        return self.current_session_id

    def log_prompt(self, prompt: Union[str, List[Dict[str, str]]],
                  tokens: Optional[int] = None, estimated_cost: Optional[float] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a prompt sent to the LLM."""
        # Format the prompt for storage
        formatted_prompt = prompt
        if isinstance(prompt, list):
            # It's a message list format
            formatted_prompt = "\n\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in prompt
            ])

        prompt_entry = {
            "type": "prompt",
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": formatted_prompt,
            "raw_prompt": prompt,
            "tokens": tokens,
            "estimated_cost": estimated_cost,
            "metadata": metadata or {}
        }

        self.entries.append(prompt_entry)

    def log_completion(self, completion: str, tokens: Optional[int] = None,
                      estimated_cost: Optional[float] = None,
                      latency: Optional[float] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a completion received from the LLM."""
        completion_entry = {
            "type": "completion",
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "completion": completion,
            "tokens": tokens,
            "estimated_cost": estimated_cost,
            "latency": latency,
            "metadata": metadata or {}
        }

        self.entries.append(completion_entry)

    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any],
                      result: Dict[str, Any], latency: Optional[float] = None,
                      cached: bool = False,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a tool call with its arguments and results."""
        tool_entry = {
            "type": "tool_call",
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "latency": latency,
            "cached": cached,
            "metadata": metadata or {}
        }

        self.entries.append(tool_entry)

    def log_reasoning(self, reasoning: str, step: Optional[int] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a reasoning step."""
        reasoning_entry = {
            "type": "reasoning",
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning,
            "step": step,
            "metadata": metadata or {}
        }

        self.entries.append(reasoning_entry)

    def log_plan(self, plan: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a plan created by a planner."""
        plan_entry = {
            "type": "plan",
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "plan": plan,
            "metadata": metadata or {}
        }

        self.entries.append(plan_entry)

    def log_error(self, error_type: str, error_message: str,
                 context: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an error that occurred during execution."""
        error_entry = {
            "type": "error",
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "metadata": metadata or {}
        }

        self.entries.append(error_entry)

    def log_custom(self, event_type: str, content: Dict[str, Any]) -> None:
        """Log a custom event with arbitrary content."""
        custom_entry = {
            "type": event_type,
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            **content
        }

        self.entries.append(custom_entry)

    def save(self, filepath: Optional[str] = None) -> str:
        """Save the audit trail to a JSON file."""
        import json
        import os

        # Use provided filepath or default storage path with generated filename
        if filepath:
            output_path = filepath
        else:
            if not self.storage_path:
                self.storage_path = "audit_trails"

            os.makedirs(self.storage_path, exist_ok=True)
            output_path = os.path.join(
                self.storage_path,
                f"audit_{self.current_session_id}.json"
            )

        with open(output_path, 'w') as f:
            json.dump({
                "session_id": self.current_session_id,
                "entries": self.entries,
                "summary": self.get_summary()
            }, f, indent=2)

        return output_path

    def clear(self) -> None:
        """Clear the audit trail entries."""
        self.entries = []

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the audit trail."""
        if not self.entries:
            return {"message": "No audit entries"}

        # Count entry types
        counts = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_estimated_cost = 0
        tool_usage = {}

        for entry in self.entries:
            entry_type = entry.get("type", "unknown")
            counts[entry_type] = counts.get(entry_type, 0) + 1

            # Sum up token usage and costs
            if entry_type == "prompt":
                total_prompt_tokens += entry.get("tokens", 0) or 0
                total_estimated_cost += entry.get("estimated_cost", 0) or 0
            elif entry_type == "completion":
                total_completion_tokens += entry.get("tokens", 0) or 0
                total_estimated_cost += entry.get("estimated_cost", 0) or 0
            elif entry_type == "tool_call":
                tool_name = entry.get("tool_name", "unknown")
                if tool_name not in tool_usage:
                    tool_usage[tool_name] = 0
                tool_usage[tool_name] += 1

        # Get time span
        if self.entries:
            start_time = self.entries[0].get("timestamp", "")
            end_time = self.entries[-1].get("timestamp", "")
        else:
            start_time = ""
            end_time = ""

        return {
            "session_id": self.current_session_id,
            "entry_counts": counts,
            "total_entries": len(self.entries),
            "token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            },
            "estimated_cost": total_estimated_cost,
            "tool_usage": tool_usage,
            "time_span": {
                "start": start_time,
                "end": end_time
            }
        }

    def get_token_usage_by_session(self) -> Dict[str, Dict[str, int]]:
        """Get token usage broken down by session."""
        usage_by_session = {}

        for entry in self.entries:
            session_id = entry.get("session_id", "unknown")
            entry_type = entry.get("type", "unknown")

            if session_id not in usage_by_session:
                usage_by_session[session_id] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }

            if entry_type == "prompt":
                tokens = entry.get("tokens", 0) or 0
                usage_by_session[session_id]["prompt_tokens"] += tokens
                usage_by_session[session_id]["total_tokens"] += tokens
            elif entry_type == "completion":
                tokens = entry.get("tokens", 0) or 0
                usage_by_session[session_id]["completion_tokens"] += tokens
                usage_by_session[session_id]["total_tokens"] += tokens

        return usage_by_session

    def get_tool_usage_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed usage statistics for each tool."""
        tool_stats = {}

        for entry in self.entries:
            if entry.get("type") != "tool_call":
                continue

            tool_name = entry.get("tool_name", "unknown")
            latency = entry.get("latency")
            cached = entry.get("cached", False)

            if tool_name not in tool_stats:
                tool_stats[tool_name] = {
                    "call_count": 0,
                    "cache_hits": 0,
                    "total_latency": 0,
                    "avg_latency": 0,
                    "arguments_used": []
                }

            stats = tool_stats[tool_name]
            stats["call_count"] += 1

            if cached:
                stats["cache_hits"] += 1

            if latency:
                stats["total_latency"] += latency
                stats["avg_latency"] = stats["total_latency"] / stats["call_count"]

            # Track argument patterns (simplified)
            args = entry.get("arguments", {})
            arg_keys = tuple(sorted(args.keys()))
            if arg_keys not in [tuple(sorted(a.keys())) for a in stats["arguments_used"]]:
                stats["arguments_used"].append(args)
