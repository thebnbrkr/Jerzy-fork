# chain.py - Execution chains for conversational workflows
from jerzy.common import *


from datetime import datetime
from typing import Callable, Dict, Any, Optional, List
from .memory import Memory, EnhancedMemory
from .core import State
from .llm import LLM

class Chain:
    """Composes multiple operations into a single workflow."""
    def __init__(self):
        self.steps = []
        self.memory = Memory()

    def add(self, step_func: Callable[[Dict[str, Any], Memory], Dict[str, Any]]) -> 'Chain':
        self.steps.append(step_func)
        return self

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        context = inputs
        for step in self.steps:
            context = step(context, self.memory)
        return context

class ConversationChain:
    """Manages a conversational flow with memory and state persistence."""

    def __init__(self, llm: LLM, memory: Optional[EnhancedMemory] = None, system_prompt: str = None):
        self.llm = llm
        self.memory = memory or EnhancedMemory()
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.state = State()

    def add_message(self, role: str, content: str, thread_id: str = "default",
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.memory.add_to_thread(thread_id, message)

    def get_conversation_context(self, thread_id: str = "default",
                                 context_window: int = 10,
                                 include_system_prompt: bool = True) -> List[Dict[str, Any]]:
        messages = []
        if include_system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        recent_history = self.memory.get_thread(thread_id, context_window)
        for entry in recent_history:
            if entry["role"] in ["user", "assistant", "system"]:
                messages.append({
                    "role": entry["role"],
                    "content": entry["content"]
                })

        return messages

    def generate_response(self, query: str, thread_id: str = "default",
                          context_window: int = 10) -> str:
        messages = self.get_conversation_context(thread_id, context_window)
        if not messages or messages[-1]["role"] != "user" or messages[-1]["content"] != query:
            messages.append({"role": "user", "content": query})
            self.add_message("user", query, thread_id)

        response = self.llm.generate(messages)
        self.add_message("assistant", response, thread_id)
        return response

    def search_and_respond(self, query: str, thread_id: str = "default",
                           context_window: int = 5, relevant_k: int = 3) -> str:
        messages = self.get_conversation_context(thread_id, context_window)
        relevant_msgs = self.memory.find_relevant(query, top_k=relevant_k)

        if relevant_msgs:
            context_msg = "Here are some relevant messages from our past conversation:\n\n"
            for msg in relevant_msgs:
                context_msg += f"{msg['role']}: {msg['content']}\n\n"
            context_msg += "Please consider this context when responding."

            if messages and messages[0]["role"] == "system":
                messages.insert(1, {"role": "system", "content": context_msg})
            else:
                messages.insert(0, {"role": "system", "content": context_msg})

        messages.append({"role": "user", "content": query})
        self.add_message("user", query, thread_id)

        response = self.llm.generate(messages)
        self.add_message("assistant", response, thread_id)
        return response

    def run(self, input_text: str, thread_id: str = "default",
            context_window: int = 10, use_search: bool = True) -> Dict[str, Any]:
        start_time = datetime.now()
        response = self.search_and_respond(input_text, thread_id, context_window) if use_search                    else self.generate_response(input_text, thread_id, context_window)
        end_time = datetime.now()
        return {
            "response": response,
            "thread_id": thread_id,
            "timestamp": end_time.isoformat(),
            "processing_time": (end_time - start_time).total_seconds(),
            "context_window": context_window,
            "used_search": use_search
        }

    def summarize_conversation(self, thread_id: str = "default") -> str:
        return self.memory.summarize_thread(thread_id, self.llm)

    def save_conversation(self, filepath: str) -> None:
        self.memory.save_to_file(filepath)

    def load_conversation(self, filepath: str) -> None:
        self.memory.load_from_file(filepath)
