# memory.py - Memory systems for conversational and reasoning context
from jerzy.common import *


from datetime import datetime
from typing import Any, Dict, List, Optional
import json


class Memory:
    """Memory store with better tracking of conversation and reasoning."""

    def __init__(self):
        self.storage = {}
        self.history = []
        self.tool_calls = []
        self.reasoning_steps = []

    def set(self, key: str, value: Any) -> None:
        """Store a value by key."""
        self.storage[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key."""
        return self.storage.get(key, default)

    def add_to_history(self, entry: Dict[str, Any]) -> None:
        """Add an entry to the conversation history."""
        # Add a timestamp if not present
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()

        self.history.append(entry)

        # Track specialized entries
        if entry.get("type") == "reasoning":
            self.reasoning_steps.append(entry)
        elif entry.get("type") == "tool_call":
            self.tool_calls.append(entry)

    def get_history(self, last_n: Optional[int] = None,
                    entry_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get conversation history, optionally filtered by entry types and limited to last n entries."""
        if entry_types:
            filtered_history = [entry for entry in self.history if entry.get("type") in entry_types]
        else:
            filtered_history = self.history

        if last_n is not None:
            return filtered_history[-last_n:]
        return filtered_history

    def get_unique_tool_results(self, tool_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get unique results from tool calls, optionally filtered by tool name."""
        seen_results = set()
        unique_results = []

        for entry in self.history:
            if entry.get("role") == "system" and "Tool result:" in entry.get("content", ""):
                if tool_name is None or tool_name in entry.get("content", ""):
                    result_content = entry.get("content", "").replace("Tool result:", "").strip()

                    # Only add if we haven't seen this result before
                    if result_content not in seen_results:
                        seen_results.add(result_content)
                        unique_results.append(result_content)

        return unique_results

    def get_last_reasoning(self) -> Optional[str]:
        """Get the most recent reasoning step."""
        for entry in reversed(self.history):
            if entry.get("type") == "reasoning":
                return entry.get("content", "").replace("Reasoning:", "").strip()
        return None

    def get_reasoning_chain(self) -> List[str]:
        """Get the full chain of reasoning steps."""
        return [entry.get("content", "").replace("Reasoning:", "").strip()
                for entry in self.history if entry.get("type") == "reasoning"]


class EnhancedMemory(Memory):
    """Enhanced memory store with conversation threading and semantic retrieval."""

    def __init__(self, max_history_length: int = 100):
        super().__init__()
        self.max_history_length = max_history_length
        self.threads = {}  # Map of thread_id -> list of message indices
        self.current_thread = "default"
        self.indexed_content = {}  # For simple keyword retrieval

    def add_to_thread(self, thread_id: str, entry: Dict[str, Any]) -> None:
        """Add an entry to a specific conversation thread."""
        if thread_id not in self.threads:
            self.threads[thread_id] = []

        # Add to global history first
        self.add_to_history(entry)

        # Track this entry in the thread
        entry_index = len(self.history) - 1
        self.threads[thread_id].append(entry_index)

        # Index content for keyword retrieval
        if "content" in entry:
            words = set(entry["content"].lower().split())
            for word in words:
                if word not in self.indexed_content:
                    self.indexed_content[word] = []
                self.indexed_content[word].append(entry_index)

    def get_thread(self, thread_id: str, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from a specific thread."""
        if thread_id not in self.threads:
            return []

        indices = self.threads[thread_id]
        if last_n is not None:
            indices = indices[-last_n:]

        return [self.history[i] for i in indices]

    def summarize_thread(self, thread_id: str, summarizer_llm=None) -> str:
        """Create a summary of the thread."""
        thread = self.get_thread(thread_id)

        if not thread:
            return "No messages in this thread."

        if summarizer_llm:
            # Format the conversation for the LLM
            conversation_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in thread
            ])

            prompt = f"""
            Please summarize the following conversation in 2-3 sentences:

            {conversation_text}

            Summary:
            """

            try:
                return summarizer_llm.generate(prompt)
            except Exception as e:
                return f"Error generating summary: {str(e)}. Thread has {len(thread)} messages."

        # Fallback if no LLM provided
        return f"Thread with {len(thread)} messages"

    def prune_history(self, keep_last_n: int = None) -> None:
        """Remove older history entries to manage context window."""
        if not keep_last_n or len(self.history) <= keep_last_n:
            return

        # Keep the most recent entries
        to_remove = len(self.history) - keep_last_n
        removed_entries = self.history[:to_remove]
        self.history = self.history[to_remove:]

        # Update thread indices
        for thread_id in self.threads:
            # Filter out removed indices and adjust remaining ones
            self.threads[thread_id] = [i - to_remove for i in self.threads[thread_id]
                                       if i >= to_remove]

        # Update indexed content
        for word in self.indexed_content:
            self.indexed_content[word] = [i - to_remove for i in self.indexed_content[word]
                                          if i >= to_remove]

    def find_relevant(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find the most relevant history entries using simple keyword matching."""
        # Split query into words for matching
        query_words = query.lower().split()

        # Find indices that contain any query word
        candidate_indices = set()
        for word in query_words:
            if word in self.indexed_content:
                candidate_indices.update(self.indexed_content[word])

        # Score each candidate by counting matching words
        scores = []
        for idx in candidate_indices:
            if idx < len(self.history):
                content = self.history[idx].get("content", "").lower()
                score = sum(1 for word in query_words if word in content)
                scores.append((idx, score))

        # Sort by relevance score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k relevant entries
        top_indices = [i for i, score in scores[:top_k] if score > 0]
        return [self.history[i] for i in top_indices]

    def save_to_file(self, filepath: str) -> None:
        """Save memory to a JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "history": self.history,
                "threads": self.threads,
                "current_thread": self.current_thread
            }, f)

    def load_from_file(self, filepath: str) -> None:
        """Load memory from a JSON file."""
        import json
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.history = data["history"]
                self.threads = data["threads"]
                self.current_thread = data.get("current_thread", "default")

                # Rebuild index
                self.indexed_content = {}
                for i, entry in enumerate(self.history):
                    if "content" in entry:
                        words = set(entry["content"].lower().split())
                        for word in words:
                            if word not in self.indexed_content:
                                self.indexed_content[word] = []
                            self.indexed_content[word].append(i)
        except Exception as e:
            print(f"Error loading memory: {str(e)}")

