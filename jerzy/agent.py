# agent.py - Central LLM-powered agent interface

from jerzy.common import *


from typing import List, Any, Optional
from .core import ToolCache, State
from .trace import Trace
from .memory import Memory
from .chain import ConversationChain
from .llm import LLM

class Agent:
    def __init__(self, llm: LLM, system_prompt: Optional[str] = None,
                 cache_ttl: Optional[int] = 3600, cache_size: int = 100,
                 enable_auditing: bool = True):
        self.llm = llm
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.tools = []
        self.memory = Memory()
        self.trace = Trace(self.memory)
        self.cache = ToolCache(max_size=cache_size, ttl=cache_ttl)
        self.state = State()
        self.tool_call_history = []

        if enable_auditing:
            self.audit_trail = None  # You can implement an audit module
        else:
            self.audit_trail = None

    def add_tools(self, tools: List[Any]) -> None:
        for tool in tools:
            if tool.name not in {t.name for t in self.tools}:
                self.tools.append(tool)

    def remember(self, key: str, value: Any) -> None:
        if not hasattr(self, 'conversation') or not self.conversation:
            self.conversation = ConversationChain(self.llm, system_prompt=self.system_prompt)
        self.conversation.memory.set(key, value)
        self.conversation.add_message("system", f"Stored information: {key} = {str(value)}", "default")

    def chat(self, user_message: str, thread_id: str = "default",
             use_semantic_search: bool = False, context_window: int = 10) -> str:
        if not hasattr(self, 'conversation') or not self.conversation:
            self.conversation = ConversationChain(self.llm, system_prompt=self.system_prompt)

        if use_semantic_search:
            return self.conversation.search_and_respond(user_message, thread_id, context_window)
        else:
            return self.conversation.generate_response(user_message, thread_id, context_window)

    def save_conversation(self, filepath: str) -> None:
        if hasattr(self, 'conversation') and self.conversation:
            self.conversation.save_conversation(filepath)

    def load_conversation(self, filepath: str) -> None:
        if not hasattr(self, 'conversation') or not self.conversation:
            self.conversation = ConversationChain(self.llm, system_prompt=self.system_prompt)
        self.conversation.load_conversation(filepath)


class EnhancedAgent(Agent):
    def __init__(self, llm: LLM, system_prompt: Optional[str] = None,
                 cache_ttl: Optional[int] = 3600, cache_size: int = 100):
        super().__init__(llm, system_prompt, cache_ttl, cache_size)
        self.state = State()
        # Planner logic can be added here
