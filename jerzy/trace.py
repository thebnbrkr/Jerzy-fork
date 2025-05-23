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
