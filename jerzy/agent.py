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



class ConversationalAgent(Agent):
    """Agent specialized for multi-turn conversational interactions."""

    def __init__(self, llm: LLM, system_prompt: Optional[str] = None,
                 cache_ttl: Optional[int] = 3600, cache_size: int = 100,
                 use_vector_memory: bool = False):
        super().__init__(llm, system_prompt, cache_ttl, cache_size)
        self.conversation = ConversationChain(
            llm,
            EnhancedMemory(),
            system_prompt or "You are a helpful assistant that remembers previous interactions."
        )

        if use_vector_memory:
            self.init_vector_memory()

    def chat(self, message: str, thread_id: str = "default",
             use_search: bool = True, context_window: int = 10) -> str:
        """Chat with the agent, maintaining conversation history."""
        # First try to use tools if needed
        if self.tools and len(self.tools) > 0:
            # Create a message format that includes history and the current message
            messages = self.conversation.get_conversation_context(thread_id, context_window)
            messages.append({"role": "user", "content": message})

            # Try to see if tools are needed
            response = self.llm.generate_with_tools(messages, self.tools)

            # Add user message to conversation history
            self.conversation.add_message("user", message, thread_id)

            # If the model wants to use a tool
            if response["type"] == "tool_call":
                tool_name = response["tool"]
                tool_args = response["args"]
                tool_reasoning = response.get("reasoning", "")

                # Find the tool
                tool = next((t for t in self.tools if t.name == tool_name), None)

                if tool:
                    # Execute the tool
                    cache_to_use = self.cache if hasattr(self, 'cache') else None
                    tool_result = tool(cache=cache_to_use, **tool_args)

                    # Record the tool call
                    self.conversation.add_message(
                        "assistant",
                        f"I'll use the {tool_name} tool with these parameters: {json.dumps(tool_args)}",
                        thread_id
                    )

                    if tool_reasoning:
                        self.conversation.add_message(
                            "system",
                            f"Tool reasoning: {tool_reasoning}",
                            thread_id,
                            {"type": "reasoning"}
                        )

                    # Record the tool result
                    cache_notice = " (from cache)" if tool_result.get("cached", False) else ""
                    result_content = f"Tool {tool_name} returned{cache_notice}: {json.dumps(tool_result.get('result', tool_result))}"

                    self.conversation.add_message("system", result_content, thread_id)

                    # Generate final answer using the tool result
                    messages = self.conversation.get_conversation_context(thread_id, context_window + 3)
                    messages.append({
                        "role": "system",
                        "content": "Based on the tool results above, provide a helpful response to the user."
                    })

                    final_response = self.llm.generate(messages)
                    self.conversation.add_message("assistant", final_response, thread_id)

                    return final_response

            # If we didn't use a tool, continue with normal conversation flow

        # Normal conversation flow (no tools needed)
        if use_search:
            return self.conversation.search_and_respond(
                message, thread_id, context_window
            )
        else:
            return self.conversation.generate_response(
                message, thread_id, context_window
            )

    def start_new_conversation(self, thread_id: str = None) -> str:
        """Start a new conversation thread."""
        if thread_id is None:
            thread_id = f"conversation_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Create the new thread
        self.conversation.memory.threads[thread_id] = []

        return thread_id

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversation threads with summary info."""
        result = []

        for thread_id, indices in self.conversation.memory.threads.items():
            # Get first and last message timestamp
            if indices:
                first_msg = self.conversation.memory.history[indices[0]]
                last_msg = self.conversation.memory.history[indices[-1]]

                first_timestamp = first_msg.get("timestamp", "")
                last_timestamp = last_msg.get("timestamp", "")

                # Get a summary
                summary = self.conversation.summarize_conversation(thread_id)

                result.append({
                    "thread_id": thread_id,
                    "message_count": len(indices),
                    "first_message": first_timestamp,
                    "last_message": last_timestamp,
                    "summary": summary
                })

        return result

    def get_conversation_history(self, thread_id: str = "default",
                                 formatted: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """Get the conversation history for a thread."""
        history = self.conversation.memory.get_thread(thread_id)

        if formatted:
            formatted_history = []
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "user":
                    formatted_history.append(f"User: {content}")
                elif role == "assistant":
                    formatted_history.append(f"Assistant: {content}")
                elif role == "system":
                    if "Tool result" in content:
                        formatted_history.append(f"[Tool Result] {content.replace('Tool result:', '')}")

            return "\n\n".join(formatted_history)

        return history




class EnhancedAgent(Agent):
    """Agent with explicit planning capabilities."""

    def __init__(self, llm, system_prompt=None, cache_ttl=3600, cache_size=100):
        super().__init__(llm, system_prompt, cache_ttl, cache_size)
        self.state = State()  # Initialize state
        self.planner = Planner(llm, self.tools, self.state)

    def plan_and_execute(self, goal: str, context: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
        """Generate and execute a plan to achieve a goal."""
        # Create a plan
        plan = self.planner.create_plan(goal, context)

        if verbose:
            actionable_steps = plan.get_actionable_steps()
            print(f"📝 Plan created with {len(actionable_steps)} actionable steps for goal: {goal}")
            print("-" * 50)
            for i, step in enumerate(actionable_steps):
                print(f"Step {i + 1}: {step['description']}")
                if step.get("tool"):
                    print(f"  Tool: {step['tool']}")
                    print(f"  Parameters: {json.dumps(step.get('params', {}))}")
                if step.get("depends_on"):
                    print(f"  Depends on: {step['depends_on']}")
                print()

            print(plan.visualize())

        # Execute the plan
        execution_result = self.planner.execute_plan(plan, verbose)

        # Store the executed plan in state
        self.state.set("plan.current", plan.to_dict())
        self.state.set("plan.execution_result", execution_result)

        # Generate a summary of the execution
        summary_prompt = f"""
        I executed a plan to achieve this goal: "{goal}"

        The steps and results were:
        {json.dumps(execution_result, indent=2)}

        Please provide a concise summary of the execution results, highlighting key findings
        and whether the goal was achieved. Focus on the actual data and insights discovered,
        not the process itself.
        """

        summary = self.llm.generate(summary_prompt)

        if verbose:
            print("\n📊 Execution Summary:")
            print("-" * 50)
            print(summary)

        return {
            "goal": goal,
            "plan": plan.to_dict(),
            "execution_result": execution_result,
            "summary": summary
        }

class AgentRole:
    """Defines a specialized role for an agent in a multi-agent system."""

    def __init__(self, name: str, description: str, system_prompt: str):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt


class AgentMessage:
    """Represents a message in a multi-agent conversation."""

    def __init__(self, sender: str, receiver: str, content: str,
                 message_type: str = "message", metadata: Dict[str, Any] = None):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type  # "message", "question", "proposal", "critique", etc.
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "type": self.message_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class MultiAgentSystem:
    """A system for coordinating multiple agents with different roles."""

    def __init__(self, llm: LLM, shared_state: State = None):
        self.llm = llm
        self.agents = {}  # {name: Agent}
        self.roles = {}  # {name: AgentRole}
        self.conversation = []  # List of AgentMessage objects
        self.state = shared_state or State()

    def add_agent(self, name: str, role: AgentRole, tools: List[Tool] = None) -> None:
        """Add an agent with a specific role to the system."""
        agent = Agent(self.llm, role.system_prompt)

        # Add tools if provided
        if tools:
            agent.add_tools(tools)

        self.agents[name] = agent
        self.roles[name] = role

    def broadcast_message(self, sender: str, content: str,
                          message_type: str = "message") -> None:
        """Send a message to all agents from the specified sender."""
        for receiver in self.agents.keys():
            if receiver != sender:
                self._send_message(sender, receiver, content, message_type)

    def _send_message(self, sender: str, receiver: str, content: str,
                      message_type: str = "message", metadata: Dict[str, Any] = None) -> None:
        """Send a message from one agent to another."""
        message = AgentMessage(sender, receiver, content, message_type, metadata)
        self.conversation.append(message)

    def _format_conversation_for_agent(self, agent_name: str) -> str:
        """Format the conversation history for a specific agent."""
        relevant_messages = []

        for msg in self.conversation:
            # Include messages where this agent is the sender or receiver
            # or broadcast messages where this agent is not the sender
            if msg.sender == agent_name or msg.receiver == agent_name or \
                    (msg.receiver == "all" and msg.sender != agent_name):
                if msg.sender == agent_name:
                    prefix = "You"
                else:
                    prefix = msg.sender

                relevant_messages.append(f"{prefix}: {msg.content}")

        return "\n\n".join(relevant_messages)

    def collaborative_solve(self, problem: str, max_turns: int = 10,
                            verbose: bool = False) -> Dict[str, Any]:
        """Have agents work together to solve a problem."""
        if not self.agents:
            raise ValueError("No agents in the system. Add agents first.")

        # Initialize the conversation with the problem
        self.conversation = []
        facilitator = "Facilitator"
        self._send_message(facilitator, "all",
                           f"Please solve this problem collaboratively: {problem}",
                           "problem")

        if verbose:
            print(f"🤝 Starting collaborative solving for problem: {problem}")
            print("-" * 50)

        # Track which agents have contributed in the current round
        for turn in range(max_turns):
            contributed = set()

            # Each agent takes a turn
            for agent_name, agent in self.agents.items():
                if agent_name in contributed:
                    continue

                # Prepare context for this agent
                context = self._format_conversation_for_agent(agent_name)
                agent_role = self.roles[agent_name]

                prompt = f"""
                You are {agent_name}, with this role: {agent_role.description}

                The group is trying to solve this problem: {problem}

                Conversation history:
                {context}

                Based on the conversation so far and your expertise, provide your contribution.
                You can:
                1. Share information or insights
                2. Ask questions to other agents
                3. Propose a solution or next step
                4. Critique or build upon others' ideas

                If you think the problem is solved, state that clearly.
                """

                # Get the agent's response
                response = agent.llm.generate(prompt)

                # Determine if this is a question, proposal, or regular message
                message_type = "message"
                if "?" in response and any(question_word in response.lower()
                                           for question_word in ["who", "what", "when", "where", "why", "how"]):
                    message_type = "question"
                elif any(proposal_word in response.lower()
                         for proposal_word in ["propose", "suggest", "recommend", "solution", "idea"]):
                    message_type = "proposal"

                # Send the message to all other agents
                self._send_message(agent_name, "all", response, message_type)
                contributed.add(agent_name)

                if verbose:
                    print(f"{agent_name}: {response}")
                    print("-" * 30)

                # Check if the agent thinks the problem is solved
                if "problem is solved" in response.lower() or "solution is complete" in response.lower():
                    # Have other agents confirm
                    confirmations = self._check_solution_consensus(problem, agent_name, response)

                    if all(confirmations.values()):
                        if verbose:
                            print("✅ All agents agree the problem is solved!")

                        # Generate final solution summary
                        return self._generate_solution_summary(problem)

            # If we've reached the maximum turns, generate a solution anyway
            if turn == max_turns - 1:
                if verbose:
                    print("⏰ Maximum turns reached. Generating solution summary.")

                return self._generate_solution_summary(problem)

        # Should not reach here if max_turns is positive
        return {"status": "error", "message": "No solution found"}

    def _check_solution_consensus(self, problem: str, proposer: str,
                                  proposed_solution: str) -> Dict[str, bool]:
        """Check if all agents agree with the proposed solution."""
        confirmations = {}

        for agent_name, agent in self.agents.items():
            if agent_name == proposer:
                confirmations[agent_name] = True
                continue

            prompt = f"""
            You are {agent_name}, with this role: {self.roles[agent_name].description}

            The original problem was: {problem}

            {proposer} has proposed this solution:
            {proposed_solution}

            Do you agree that this solution adequately solves the problem?
            Answer YES if you agree, or NO followed by your specific objections if you disagree.
            """

            response = agent.llm.generate(prompt)
            agrees = response.strip().upper().startswith("YES")

            confirmations[agent_name] = agrees

            # Add the confirmation or objection to the conversation
            message_type = "confirmation" if agrees else "objection"
            self._send_message(agent_name, "all", response, message_type)

        return confirmations

    def _generate_solution_summary(self, problem: str) -> Dict[str, Any]:
        """Generate a summary of the solution based on the conversation."""
        # Create a summary prompt for the LLM
        conversation_text = "\n\n".join([
            f"{msg.sender}: {msg.content}" for msg in self.conversation
        ])

        summary_prompt = f"""
        The following is a conversation between multiple agents trying to solve this problem:

        PROBLEM: {problem}

        CONVERSATION:
        {conversation_text}

        Please provide:
        1. A concise summary of the final solution
        2. The key contributions from each agent
        3. Any areas of disagreement or uncertainty

        Format your response as a structured report suitable for presentation.
        """

        # Use the LLM to generate a summary
        summary = self.llm.generate(summary_prompt)

        return {
            "problem": problem,
            "conversation": [msg.to_dict() for msg in self.conversation],
            "summary": summary
        }

    def debate(self, topic: str, rounds: int = 3, verbose: bool = False) -> Dict[str, Any]:
        """Have agents debate a topic with structured rounds."""
        if not self.agents:
            raise ValueError("No agents in the system. Add agents first.")
        if len(self.agents) < 2:
            raise ValueError("Debate requires at least two agents.")

        # Initialize the conversation with the topic
        self.conversation = []
        facilitator = "Facilitator"
        self._send_message(facilitator, "all",
                           f"Please debate this topic: {topic}",
                           "topic")

        if verbose:
            print(f"🎭 Starting debate on topic: {topic}")
            print("-" * 50)

        # Initial positions
        for agent_name, agent in self.agents.items():
            role = self.roles[agent_name]

            prompt = f"""
            You are {agent_name}, with this role: {role.description}

            The topic for debate is: {topic}

            Provide your initial position on this topic. Be clear and concise,
            focusing on your strongest 2-3 arguments.
            """

            position = agent.llm.generate(prompt)
            self._send_message(agent_name, "all", position, "position")

            if verbose:
                print(f"{agent_name} (Initial Position): {position}")
                print("-" * 30)

        # Debate rounds
        for round_num in range(rounds):
            if verbose:
                print(f"\nRound {round_num + 1}:")
                print("-" * 30)

            for agent_name, agent in self.agents.items():
                # Prepare context with all positions so far
                context = self._format_conversation_for_agent(agent_name)

                prompt = f"""
                You are {agent_name}, with this role: {self.roles[agent_name].description}

                The topic for debate is: {topic}

                This is round {round_num + 1} of the debate.

                Previous discussion:
                {context}

                Based on the discussion so far, provide your response. You should:
                1. Address the strongest counterarguments to your position
                2. Strengthen your main points with new evidence or reasoning
                3. Find points of agreement where possible
                """

                response = agent.llm.generate(prompt)
                self._send_message(agent_name, "all", response, "debate_round")

                if verbose:
                    print(f"{agent_name}: {response}")
                    print("-" * 30)

        # Have a neutral summarizer create a final summary
        conclusion_prompt = f"""
        The following is a debate between multiple perspectives on this topic:

        TOPIC: {topic}

        DEBATE TRANSCRIPT:
        {self._format_conversation_for_agent("summarizer")}

        As a neutral summarizer, please:
        1. Summarize the key points made by each side
        2. Identify the strongest arguments presented
        3. Note any areas of consensus or compromise
        4. Evaluate the quality of evidence and reasoning used

        Provide a balanced assessment that doesn't favor any particular position.
        """

        conclusion = self.llm.generate(conclusion_prompt)

        if verbose:
            print("\n📝 Debate Conclusion:")
            print(conclusion)

        return {
            "topic": topic,
            "rounds": rounds,
            "participants": list(self.agents.keys()),
            "conversation": [msg.to_dict() for msg in self.conversation],
            "conclusion": conclusion
        }

