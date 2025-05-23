# llm.py - Language model interfaces and implementations
from jerzy.common import *


from datetime import datetime
from typing import List, Dict, Any, Optional
from .core import Tool
import json
import re

########


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
        """Generate a response for the given prompt."""
        raise NotImplementedError("Subclasses must implement generate()")

    def generate_with_tools(self, prompt: str, tools: List[Tool]) -> Dict[str, Any]:
        """Generate a response that might include tool calls."""
        raise NotImplementedError("Subclasses must implement generate_with_tools()")

    def get_token_usage(self) -> Dict[str, int]:
        """Get the current token usage statistics."""
        return self.token_usage

    def get_token_usage_history(self) -> List[Dict[str, Any]]:
        """Get the history of token usage per request."""
        return self.token_usage_history

    def reset_token_usage(self) -> None:
        """Reset token usage statistics."""
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
            raise ImportError("OpenAI package not installed. Install it with 'pip install openai'")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate a text response using the OpenAI API."""
        # Handle both string and message list formats
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        # Track token usage from the response
        if hasattr(response, 'usage'):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "timestamp": datetime.now().isoformat()
            }

            # Some providers include estimated cost
            if hasattr(response.usage, 'estimated_cost'):
                usage["estimated_cost"] = response.usage.estimated_cost

            # Add to cumulative totals
            self.token_usage["prompt_tokens"] += usage["prompt_tokens"]
            self.token_usage["completion_tokens"] += usage["completion_tokens"]
            self.token_usage["total_tokens"] += usage["total_tokens"]
            if "estimated_cost" in usage:
                self.token_usage["estimated_cost"] += usage["estimated_cost"]

            # Store in history
            self.token_usage_history.append(usage)

        return response.choices[0].message.content

    def generate_with_tools(self, prompt, tools: List[Tool], reasoning_mode: str = "medium") -> Dict[str, Any]:
        """Generate a response that might include tool calls using OpenAI function calling.

        Args:
            prompt: Text prompt or list of message dictionaries
            tools: List of available tools
            reasoning_mode: Controls verbosity of reasoning ("none", "short", "medium", "full")
        """
        # Handle both string and message list formats
        if isinstance(prompt, list):
            messages = prompt
        elif isinstance(prompt, str):
            # Parse string format into messages
            lines = prompt.split('\n')
            messages = []
            current_role = None
            current_content = []

            for line in lines:
                if line.startswith('system:') or line.startswith('user:') or line.startswith('assistant:'):
                    # Save the previous message if there is one
                    if current_role is not None:
                        messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
                        current_content = []

                    # Start a new message
                    parts = line.split(':', 1)
                    current_role = parts[0].strip()
                    if len(parts) > 1:
                        current_content.append(parts[1].strip())
                else:
                    # Continue the current message
                    if current_role is not None:
                        current_content.append(line)

            # Add the last message
            if current_role is not None:
                messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})

            # If we couldn't parse any messages, use the entire prompt as a user message
            if not messages:
                messages = [{"role": "user", "content": prompt}]
        else:
            # If it's neither a list nor a string, convert to string and use as user message
            messages = [{"role": "user", "content": str(prompt)}]

        try:
            # Try using OpenAI's function calling interface
            # Convert tool definitions to the OpenAI format
            tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                param: {"type": info["type"].lower() if hasattr(info["type"], "lower") else str(
                                    info["type"])}
                                for param, info in tool.signature.items()
                            },
                            "required": [
                                param for param, info in tool.signature.items()
                                if info.get("required", True)
                            ]
                        }
                    }
                } for tool in tools
            ]

            # Add bypass_cache parameter to tools that are cacheable
            for i, tool in enumerate(tools):
                if tool.cacheable:
                    tool_schemas[i]["function"]["parameters"]["properties"]["bypass_cache"] = {
                        "type": "boolean",
                        "description": "Set to true to bypass the cache and force a fresh call"
                    }

            # Make the API call with tool calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto"
            )

            # Track token usage from the response
            if hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "timestamp": datetime.now().isoformat()
                }

                # Some providers include estimated cost
                if hasattr(response.usage, 'estimated_cost'):
                    usage["estimated_cost"] = response.usage.estimated_cost

                # Add to cumulative totals
                self.token_usage["prompt_tokens"] += usage["prompt_tokens"]
                self.token_usage["completion_tokens"] += usage["completion_tokens"]
                self.token_usage["total_tokens"] += usage["total_tokens"]
                if "estimated_cost" in usage:
                    self.token_usage["estimated_cost"] += usage["estimated_cost"]

                # Store in history
                self.token_usage_history.append(usage)

            message = response.choices[0].message

            # Check if the model wants to call a tool
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                tool_name = tool_call.function.name

                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                # Find the matching tool
                called_tool = next((tool for tool in tools if tool.name == tool_name), None)

                if called_tool:
                    # Define reasoning prompts based on verbosity level
                    tool_reasoning_prompts = {
                        "none": None,  # No reasoning
                        "short": [
                            {"role": "system", "content": "In 1-2 sentences, explain why you chose this tool."},
                            {"role": "user", "content": f"You chose the {tool_name} tool. Why?"}
                        ],
                        "medium": [
                            {"role": "system",
                             "content": "In 3-5 sentences, explain your tool choice and expectations."},
                            {"role": "user",
                             "content": f"You chose to use {tool_name} with params: {json.dumps(arguments)[:100]}. Why this tool and what do you expect?"}
                        ],
                        "full": [
                            {"role": "system",
                             "content": "Explain your reasoning for choosing this tool and what you expect to learn from it."},
                            {"role": "user",
                             "content": f"You chose to use the {tool_name} tool with these parameters: {json.dumps(arguments)}. Why did you choose this tool and what do you expect to learn?"}
                        ]
                    }

                    # Get reasoning based on verbosity level (or empty string if none)
                    reasoning = ""
                    if reasoning_mode != "none" and reasoning_mode in tool_reasoning_prompts:
                        reasoning_prompt = tool_reasoning_prompts.get(reasoning_mode)
                        if reasoning_prompt:
                            reasoning = self.generate(reasoning_prompt)

                    # Execute the tool with the provided arguments
                    tool_result = called_tool(**arguments)

                    return {
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": arguments,
                        "result": tool_result,
                        "reasoning": reasoning,
                        "reasoning_mode": reasoning_mode
                    }

            # If no tool call or tool not found, return the text response
            return {
                "type": "text",
                "content": message.content
            }

        except Exception as e:
            # If function calling fails, fall back to the text-based approach
            # This makes our implementation model-agnostic
            return self._text_based_tool_calling(messages, tools, reasoning_mode)

    def _text_based_tool_calling(self, messages, tools, reasoning_mode: str = "medium"):
        """Fall back to text-based tool calling when function calling is not supported."""
        # Add tool descriptions to the system message
        tools_desc = "\n".join([
            f"Tool: {tool.name}\nDescription: {tool.description}\nParameters: {json.dumps(tool.signature)}\nCacheable: {tool.cacheable}"
            for tool in tools
        ])

        tool_instructions = f"""
Available Tools:
{tools_desc}

To use a tool, respond in the following format:

USE TOOL: <tool_name>
PARAMETERS:
{{
  "param1": "value1",
  "param2": "value2",
  "bypass_cache": false  # Optional: Set to true to force a fresh tool call
}}

Only use one of the tools listed above, and only when necessary. If you don't need to use a tool, just respond normally.
"""

        # Find the system message, or add one if it doesn't exist
        system_message_found = False
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                # Append tool descriptions to existing system message
                messages[i]["content"] = msg["content"] + "\n\n" + tool_instructions
                system_message_found = True
                break

        if not system_message_found:
            # Insert a new system message at the beginning
            messages.insert(0, {
                "role": "system",
                "content": f"You are a helpful assistant with access to tools.\n\n{tool_instructions}"
            })

        # Make the API call without tool calling
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        # Track token usage from the response
        if hasattr(response, 'usage'):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "timestamp": datetime.now().isoformat()
            }

            # Some providers include estimated cost
            if hasattr(response.usage, 'estimated_cost'):
                usage["estimated_cost"] = response.usage.estimated_cost

            # Add to cumulative totals
            self.token_usage["prompt_tokens"] += usage["prompt_tokens"]
            self.token_usage["completion_tokens"] += usage["completion_tokens"]
            self.token_usage["total_tokens"] += usage["total_tokens"]
            if "estimated_cost" in usage:
                self.token_usage["estimated_cost"] += usage["estimated_cost"]

            # Store in history
            self.token_usage_history.append(usage)

        message_content = response.choices[0].message.content

        # Try to parse a tool call from the response
        tool_call = self._parse_tool_call_from_text(message_content)

        if tool_call:
            tool_name = tool_call.get("tool")
            arguments = tool_call.get("args", {})

            # Find the matching tool
            called_tool = next((tool for tool in tools if tool.name == tool_name), None)

            if called_tool:
                # Define reasoning prompts based on verbosity level (same as above)
                tool_reasoning_prompts = {
                    "none": None,
                    "short": [
                        {"role": "system", "content": "In 1-2 sentences, explain why you chose this tool."},
                        {"role": "user", "content": f"You chose the {tool_name} tool. Why?"}
                    ],
                    "medium": [
                        {"role": "system", "content": "In 3-5 sentences, explain your tool choice and expectations."},
                        {"role": "user",
                         "content": f"You chose to use {tool_name} with params: {json.dumps(arguments)[:100]}. Why this tool and what do you expect?"}
                    ],
                    "full": [
                        {"role": "system",
                         "content": "Explain your reasoning for choosing this tool and what you expect to learn from it."},
                        {"role": "user",
                         "content": f"You chose to use the {tool_name} tool with these parameters: {json.dumps(arguments)}. Why did you choose this tool and what do you expect to learn?"}
                    ]
                }

                # Get reasoning based on verbosity level (or empty string if none)
                reasoning = ""
                if reasoning_mode != "none" and reasoning_mode in tool_reasoning_prompts:
                    reasoning_prompt = tool_reasoning_prompts.get(reasoning_mode)
                    if reasoning_prompt:
                        reasoning = self.generate(reasoning_prompt)

                # Execute the tool with the provided arguments
                try:
                    tool_result = called_tool(**arguments)
                    return {
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": arguments,
                        "result": tool_result,
                        "reasoning": reasoning,
                        "reasoning_mode": reasoning_mode
                    }
                except Exception as e:
                    # If tool execution fails, return an error
                    return {
                        "type": "error",
                        "content": f"I tried to use the {tool_name} tool but encountered an error: {str(e)}",
                        "error": str(e),
                        "tool": tool_name,
                        "args": arguments
                    }

        # If no tool call detected or parsing failed, return the text response
        return {
            "type": "text",
            "content": message_content
        }

    def _parse_tool_call_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to parse a tool call from text response."""
        # Look for the tool call format
        tool_pattern = r'USE TOOL: (\w+)\s+PARAMETERS:\s+({[\s\S]*?})'
        match = re.search(tool_pattern, text, re.IGNORECASE)

        if match:
            tool_name = match.group(1).strip()
            args_str = match.group(2).strip()

            try:
                args = json.loads(args_str)
                return {
                    "tool": tool_name,
                    "args": args
                }
            except json.JSONDecodeError:
                # Failed to parse JSON
                return None

        return None


# For backward compatibility with the pasted code
CustomOpenAILLM = OpenAILLM

