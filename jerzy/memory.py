# memory.py - Memory systems for conversational and reasoning context

from datetime import datetime
from typing import Any, Dict, List, Optional
import json

class Memory:
    """Basic memory store with history tracking."""

    def __init__(self):
        self.storage = {}
        self.history = []
        self.tool_calls = []
        self.reasoning_steps = []

    def set(self, key: str, value: Any) -> None:
        self.storage[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.storage.get(key, default)

    def add_to_history(self, entry: Dict[str, Any]) -> None:
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()

        self.history.append(entry)

        if entry.get("type") == "reasoning":
            self.reasoning_steps.append(entry)
        elif entry.get("type") == "tool_call":
            self.tool_calls.append(entry)

    def get_history(self, last_n: Optional[int] = None, entry_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if entry_types:
            filtered_history = [entry for entry in self.history if entry.get("type") in entry_types]
        else:
            filtered_history = self.history

        if last_n is not None:
            return filtered_history[-last_n:]
        return filtered_history

    def get_unique_tool_results(self, tool_name: Optional[str] = None) -> List[Dict[str, Any]]:
        seen_results = set()
        unique_results = []

        for entry in self.history:
            if entry.get("role") == "system" and "Tool result:" in entry.get("content", ""):
                if tool_name is None or tool_name in entry.get("content", ""):
                    result_content = entry.get("content", "").replace("Tool result:", "").strip()
                    if result_content not in seen_results:
                        seen_results.add(result_content)
                        unique_results.append(result_content)
        return unique_results

    def get_last_reasoning(self) -> Optional[str]:
        for entry in reversed(self.history):
            if entry.get("type") == "reasoning":
                return entry.get("content", "").replace("Reasoning:", "").strip()
        return None

    def get_reasoning_chain(self) -> List[str]:
        return [entry.get("content", "").replace("Reasoning:", "").strip()
                for entry in self.history if entry.get("type") == "reasoning"]


class EnhancedMemory(Memory):
    """Memory with thread management and keyword-based relevance search."""

    def __init__(self, max_history_length: int = 100):
        super().__init__()
        self.max_history_length = max_history_length
        self.threads = {}
        self.current_thread = "default"
        self.indexed_content = {}

    def add_to_thread(self, thread_id: str, entry: Dict[str, Any]) -> None:
        if thread_id not in self.threads:
            self.threads[thread_id] = []

        self.add_to_history(entry)
        entry_index = len(self.history) - 1
        self.threads[thread_id].append(entry_index)

        if "content" in entry:
            words = set(entry["content"].lower().split())
            for word in words:
                if word not in self.indexed_content:
                    self.indexed_content[word] = []
                self.indexed_content[word].append(entry_index)

    def get_thread(self, thread_id: str, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        if thread_id not in self.threads:
            return []

        indices = self.threads[thread_id]
        if last_n is not None:
            indices = indices[-last_n:]

        return [self.history[i] for i in indices]

    def summarize_thread(self, thread_id: str, summarizer_llm=None) -> str:
        thread = self.get_thread(thread_id)
        if not thread:
            return "No messages in this thread."

        if summarizer_llm:
            conversation_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in thread
            ])
            prompt = f"Please summarize the following conversation:\n\n{conversation_text}\n\nSummary:"
            try:
                return summarizer_llm.generate(prompt)
            except Exception as e:
                return f"Error generating summary: {str(e)}. Thread has {len(thread)} messages."
        return f"Thread with {len(thread)} messages"

    def prune_history(self, keep_last_n: int = None) -> None:
        if not keep_last_n or len(self.history) <= keep_last_n:
            return

        to_remove = len(self.history) - keep_last_n
        self.history = self.history[to_remove:]

        for thread_id in self.threads:
            self.threads[thread_id] = [i - to_remove for i in self.threads[thread_id] if i >= to_remove]

        for word in self.indexed_content:
            self.indexed_content[word] = [i - to_remove for i in self.indexed_content[word] if i >= to_remove]

    def find_relevant(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_words = query.lower().split()
        candidate_indices = set()
        for word in query_words:
            if word in self.indexed_content:
                candidate_indices.update(self.indexed_content[word])

        scores = []
        for idx in candidate_indices:
            if idx < len(self.history):
                content = self.history[idx].get("content", "").lower()
                score = sum(1 for word in query_words if word in content)
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, score in scores[:top_k] if score > 0]
        return [self.history[i] for i in top_indices]

    def save_to_file(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump({
                "history": self.history,
                "threads": self.threads,
                "current_thread": self.current_thread
            }, f)

    def load_from_file(self, filepath: str) -> None:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.history = data["history"]
                self.threads = data["threads"]
                self.current_thread = data.get("current_thread", "default")
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
