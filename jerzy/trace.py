# trace.py - Execution tracing for reasoning and tool usage

from jerzy.common import *


import json
from typing import Optional, Union, List, Dict, Any
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


###### Plan,and planner######

class Plan:
    """Represents a structured plan with steps, dependencies, and status tracking."""

    def __init__(self, goal: str, steps: List[Dict[str, Any]] = None):
        self.goal = goal
        self.steps = steps or []
        self.current_step_index = 0
        self.status = "planned"  # "planned", "in_progress", "completed", "failed"
        self.creation_time = datetime.now().isoformat()
        self.completion_time = None

    def add_step(self, description: str, tool: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None,
                 depends_on: List[int] = None) -> int:
        """Add a step to the plan and return its index."""
        step_index = len(self.steps)
        self.steps.append({
            "index": step_index,
            "description": description,
            "tool": tool,
            "params": params or {},
            "depends_on": depends_on or [],
            "status": "pending",  # "pending", "in_progress", "completed", "failed", "skipped"
            "result": None,
            "start_time": None,
            "end_time": None
        })
        return step_index

    def get_next_executable_step(self) -> Optional[Dict[str, Any]]:
        """Get the next step that can be executed based on dependencies."""
        for step in self.steps:
            if step["status"] == "pending":
                # Check if all dependencies are satisfied
                deps_satisfied = all(
                    self.steps[dep_idx]["status"] == "completed"
                    for dep_idx in step["depends_on"]
                )
                if deps_satisfied:
                    return step
        return None

    def update_step_status(self, step_index: int, status: str, result: Any = None) -> None:
        """Update the status and result of a step."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["status"] = status
            if result:
                self.steps[step_index]["result"] = result

            if status == "in_progress":
                self.steps[step_index]["start_time"] = datetime.now().isoformat()
            elif status in ["completed", "failed", "skipped"]:
                self.steps[step_index]["end_time"] = datetime.now().isoformat()

            # Check if plan is completed
            if all(step["status"] in ["completed", "skipped", "failed"] for step in self.steps):
                self.status = "completed" if all(
                    step["status"] in ["completed", "skipped"] for step in self.steps
                ) else "failed"
                self.completion_time = datetime.now().isoformat()

    def get_actionable_steps(self) -> List[Dict[str, Any]]:
        """Get only the steps that are actually executable (not JSON artifacts)."""
        actionable_steps = []

        for step in self.steps:
            # A step is actionable if it has a meaningful description AND
            # either has a tool to call OR is an analysis step
            if (step.get("description") and
                    (step.get("tool") or
                     # Consider steps with no tool but a description as analysis steps
                     (not step.get("tool") and len(step.get("description", "")) > 10))):
                actionable_steps.append(step)

        return actionable_steps

    def find_step_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Find a step by its index."""
        for step in self.steps:
            if step.get("index") == index:
                return step
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the plan to a dictionary for storage or visualization."""
        return {
            "goal": self.goal,
            "steps": self.steps,
            "status": self.status,
            "creation_time": self.creation_time,
            "completion_time": self.completion_time,
            "current_step_index": self.current_step_index
        }

    def visualize(self, format: str = "mermaid") -> str:
        """Visualize the plan in various formats."""
        if format == "mermaid":
            # Generate a mermaid graph representation
            lines = ["graph TD"]

            # Only visualize actionable steps
            actionable_steps = self.get_actionable_steps()

            # Add nodes for each actionable step
            for step in actionable_steps:
                step_id = f"step{step['index']}"
                status_color = {
                    "pending": "",
                    "in_progress": "::running",
                    "completed": "::success",
                    "failed": "::failure",
                    "skipped": "::skipped"
                }.get(step["status"], "")

                # Truncate long descriptions for better visualization
                desc = step["description"]
                if len(desc) > 40:
                    desc = desc[:37] + "..."

                lines.append(f"    {step_id}[\"{desc}\"{status_color}]")

                # Add edges for dependencies (only for actionable steps)
                for dep_idx in step["depends_on"]:
                    # Check if dependency is an actionable step
                    if any(s["index"] == dep_idx for s in actionable_steps):
                        dep_id = f"step{dep_idx}"
                        lines.append(f"    {dep_id} --> {step_id}")

            # Add styles
            lines.append("    classDef running fill:#ffff99;")
            lines.append("    classDef success fill:#99ff99;")
            lines.append("    classDef failure fill:#ff9999;")
            lines.append("    classDef skipped fill:#dddddd;")

            return "\n".join(lines)
        else:
            return f"Plan visualization format '{format}' not supported"


class Planner:
    """Generates and manages plans for achieving complex goals."""

    def __init__(self, llm: LLM, tools: List[Tool], state: State):
        self.llm = llm
        self.tools = tools
        self.state = state
        self.tool_map = {tool.name: tool for tool in tools}

    def create_plan(self, goal: str, context: Optional[str] = None) -> Plan:
        """Generate a structured plan to achieve the given goal."""

        # Prepare tools information for the LLM
        tools_desc = "\n".join([
            f"Tool: {tool.name}\nDescription: {tool.description}\nParameters: {json.dumps(tool.signature)}"
            for tool in self.tools
        ])

        # Create the prompt for plan generation
        prompt = f"""
        Your task is to create a detailed, step-by-step plan to achieve this goal:

        GOAL: {goal}

        {f'CONTEXT: {context}' if context else ''}

        Available tools:
        {tools_desc}

        Generate a complete plan with the following:
        1. A sequential list of steps
        2. For each step that requires a tool, specify which tool to use and what parameters to pass
        3. For each step, list any dependencies (which steps must be completed before this one)

        IMPORTANT: For tool steps, include explicit parameters that the tool needs. If a step depends on data from a previous step,
        use the format $result.STEP_INDEX.path.to.value to reference specific data from previous steps.

        Format your response as a JSON object with the following structure:
        {{
            "plan": [
                {{
                    "description": "Step description",
                    "tool": "tool_name",  # Optional, only if a tool is needed
                    "params": {{"param1": "value1"}},  # Optional, only if a tool is needed
                    "depends_on": [0, 1]  # Indices of steps this depends on, optional
                }},
                ...
            ]
        }}

        DO NOT include JSON artifacts or formatting elements as steps. Each step should be a concrete action or analysis.
        """

        # Generate the plan using the LLM
        response = self.llm.generate(prompt)

        # Parse the JSON response
        try:
            # Extract the JSON object from the response
            # Sometimes LLMs include explanatory text before/after the JSON
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                plan_json = json_match.group(1)
                plan_data = json.loads(plan_json)
            else:
                plan_data = json.loads(response)

            if "plan" not in plan_data:
                raise ValueError("Response does not contain a 'plan' key")

            # Create the plan object
            plan = Plan(goal=goal)

            # Add steps from the parsed data
            for step_data in plan_data["plan"]:
                # Skip steps that are just JSON artifacts
                if not step_data.get("description") or len(step_data.get("description", "")) < 5:
                    continue

                plan.add_step(
                    description=step_data["description"],
                    tool=step_data.get("tool"),
                    params=step_data.get("params", {}),
                    depends_on=step_data.get("depends_on", [])
                )

            return plan

        except (json.JSONDecodeError, ValueError) as e:
            # If the response is not valid JSON, try to parse it as best as we can
            print(f"Failed to parse JSON: {str(e)}. Using fallback parsing.")

            fallback_plan = Plan(goal=goal)

            # Try to extract steps with regex
            step_pattern = r'Step (\d+):\s*(.*?)(?=Step \d+:|$)'
            steps = re.findall(step_pattern, response, re.DOTALL)

            if not steps:
                # If regex fails, use simple line parsing
                lines = [line.strip() for line in response.split("\n") if line.strip()]
                for i, line in enumerate(lines):
                    if line.startswith(("Step", "1.", "2.", "3.")) and len(line) > 10:
                        fallback_plan.add_step(description=line)
            else:
                for _, step_desc in steps:
                    if len(step_desc.strip()) > 10:
                        fallback_plan.add_step(description=step_desc.strip())

            return fallback_plan

    def execute_plan(self, plan: Plan, verbose: bool = False) -> Dict[str, Any]:
        """Execute a plan with proper parameter resolution and data flow."""
        if verbose:
            print(f"🚀 Executing plan with {len(plan.get_actionable_steps())} actionable steps")
            print("-" * 50)

        # Store results for each step
        results = {}  # Map of step index to result

        # Execute only actionable steps
        for step in plan.get_actionable_steps():
            step_index = step["index"]

            if verbose:
                print(f"\n▶️ Step {step_index + 1}: {step['description']}")

            # Update step status to in_progress
            plan.update_step_status(step_index, "in_progress")

            # Check dependencies
            dependencies_met = True
            for dep_idx in step.get("depends_on", []):
                dep_step = plan.find_step_by_index(dep_idx)
                if dep_step is None or dep_step.get("status") != "completed":
                    dependencies_met = False
                    if verbose:
                        print(f"⚠️ Dependency (step {dep_idx + 1}) not met")
                    break

            if not dependencies_met:
                plan.update_step_status(step_index, "skipped")
                continue

            # Handle different types of steps
            if not step.get("tool"):
                # This is an analysis step (no tool, just reasoning)
                if verbose:
                    print("📝 Analysis step")

                # For analysis steps, generate a summary based on previous results
                prev_results_str = json.dumps({
                    f"Step {idx + 1}": result
                    for idx, result in results.items()
                }, indent=2)

                analysis_prompt = f"""
                Based on the following results from previous steps:
                {prev_results_str}

                Please provide an analysis or insight for this step:
                {step['description']}

                Give a detailed and specific response based on the data.
                """

                try:
                    analysis_result = self.llm.generate(analysis_prompt)
                    result = {
                        "status": "success",
                        "result": analysis_result,
                        "type": "analysis"
                    }
                    results[step_index] = result
                    plan.update_step_status(step_index, "completed", result)

                    if verbose:
                        print(f"✅ Analysis: {analysis_result[:100]}...")
                except Exception as e:
                    if verbose:
                        print(f"❌ Analysis failed: {str(e)}")
                    plan.update_step_status(step_index, "failed", {"error": str(e)})

                continue

            # This is a tool step - need to execute a tool
            tool_name = step["tool"]
            raw_params = step.get("params", {})

            if verbose:
                print(f"🔧 Tool: {tool_name}")
                print(f"📝 Raw parameters: {json.dumps(raw_params)}")

            # Resolve parameters that reference previous results
            resolved_params = self._resolve_parameters(raw_params, results, verbose)

            if verbose:
                print(f"📊 Resolved parameters: {json.dumps(resolved_params)}")

            # Find the tool
            tool = next((t for t in self.tools if t.name == tool_name), None)

            if not tool:
                if verbose:
                    print(f"❌ Tool not found: {tool_name}")
                error_result = {"status": "error", "error": f"Tool {tool_name} not found"}
                results[step_index] = error_result
                plan.update_step_status(step_index, "failed", error_result)
                continue

            # Execute the tool with resolved parameters
            try:
                result = tool(**resolved_params)
                results[step_index] = result

                # Store the result in state for future reference
                self.state.set(f"tools.called.{tool_name}:{json.dumps(resolved_params, sort_keys=True)}", {
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                    "step": step_index
                })

                if verbose:
                    if isinstance(result, dict):
                        result_str = json.dumps(result)
                        if len(result_str) > 200:
                            result_str = result_str[:197] + "..."
                        print(f"✅ Result: {result_str}")
                    else:
                        print(f"✅ Result: {str(result)[:200]}...")

                plan.update_step_status(step_index, "completed", result)
            except Exception as e:
                if verbose:
                    print(f"❌ Error: {str(e)}")
                error_result = {"status": "error", "error": str(e)}
                results[step_index] = error_result
                plan.update_step_status(step_index, "failed", error_result)

        # Check if plan is completed or failed
        if all(s["status"] in ["completed", "skipped"] for s in plan.get_actionable_steps()):
            plan.status = "completed"
            if verbose:
                print("\n✅ Plan completed successfully")
        else:
            plan.status = "failed"
            if verbose:
                print("\n❌ Plan failed")

        plan.completion_time = datetime.now().isoformat()

        # Store the complete results in state
        self.state.set("plan.results", results)

        return {
            "status": plan.status,
            "results": results,
            "steps": [step for step in plan.steps],
            "completion_time": plan.completion_time
        }

    def _resolve_parameters(self, params: Dict[str, Any], results: Dict[int, Any], verbose: bool = False) -> Dict[
        str, Any]:
        """Resolve parameters by substituting references to previous results."""
        resolved = {}

        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$result."):
                # This is a reference to a previous result - parse it
                try:
                    # Format is $result.STEP_INDEX.path.to.value
                    parts = value[8:].split(".")
                    step_idx = int(parts[0])

                    if step_idx in results:
                        # Get the result from the specified step
                        current = results[step_idx]

                        # Navigate through the path
                        for path_part in parts[1:]:
                            if isinstance(current, dict) and path_part in current:
                                current = current[path_part]
                            else:
                                if verbose:
                                    print(f"⚠️ Path '{path_part}' not found in result")
                                current = None
                                break

                        resolved[key] = current
                    else:
                        if verbose:
                            print(f"⚠️ Step {step_idx} result not found")
                        resolved[key] = None
                except (ValueError, IndexError) as e:
                    if verbose:
                        print(f"⚠️ Error parsing result reference '{value}': {str(e)}")
                    resolved[key] = None
            else:
                # Not a reference, use as is
                resolved[key] = value

        return resolved


