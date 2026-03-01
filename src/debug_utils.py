"""
debug_utils.py

This module provides debugging utilities for the Agentic Workflow application.
It includes capabilities for verbose debug mode, conversation state export,
message history inspection, and replay functionality for debugging and analysis.

The debug utilities integrate with the tracing and observability modules to
provide comprehensive debugging capabilities without modifying core logic.
"""

import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path


class DebugMode:
    """
    Debug mode configuration and utilities.
    
    This class manages debug mode settings and provides utilities for
    verbose output, state capture, and debugging assistance.
    
    Attributes:
        enabled (bool): Whether debug mode is enabled.
        verbose (bool): Whether verbose output is enabled.
        capture_messages (bool): Whether to capture full message history.
        capture_tool_outputs (bool): Whether to capture full tool outputs.
        output_dir (str): Directory for debug output files.
    """
    
    def __init__(
        self,
        enabled: bool = False,
        verbose: bool = False,
        capture_messages: bool = True,
        capture_tool_outputs: bool = True,
        output_dir: str = "logs/debug"
    ):
        """
        Initialize debug mode.
        
        Args:
            enabled (bool): Enable debug mode.
            verbose (bool): Enable verbose output.
            capture_messages (bool): Capture message history.
            capture_tool_outputs (bool): Capture tool outputs.
            output_dir (str): Directory for debug files.
        """
        self.enabled = enabled
        self.verbose = verbose
        self.capture_messages = capture_messages
        self.capture_tool_outputs = capture_tool_outputs
        self.output_dir = output_dir
        
        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def should_capture(self) -> bool:
        """
        Check if debug capture is enabled.
        
        Returns:
            bool: True if capture is enabled.
        """
        return self.enabled
    
    def should_print_verbose(self) -> bool:
        """
        Check if verbose printing is enabled.
        
        Returns:
            bool: True if verbose output is enabled.
        """
        return self.enabled and self.verbose
    
    def print_debug(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Print a debug message if verbose mode is enabled.
        
        Args:
            message (str): The debug message.
            data (Optional[Dict[str, Any]]): Additional data to print.
        """
        if not self.should_print_verbose():
            return
        
        timestamp = datetime.now().isoformat()
        print(f"[DEBUG] [{timestamp}] {message}")
        if data:
            print(f"  Data: {json.dumps(data, indent=2, default=str)}")
    
    def save_debug_file(self, filename: str, content: Dict[str, Any]) -> str:
        """
        Save debug content to a file.
        
        Args:
            filename (str): Name of the file.
            content (Dict[str, Any]): Content to save.
            
        Returns:
            str: Path to the saved file.
        """
        if not self.enabled:
            return ""
        
        filepath = Path(self.output_dir) / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, default=str)
        
        return str(filepath)


class ConversationState:
    """
    Represents the state of a conversation for debugging and replay.
    
    This class captures the complete state of an agent conversation,
    including all messages, tool calls, and intermediate states.
    
    Attributes:
        conversation_id (str): Unique identifier for this conversation.
        messages (List[Dict[str, Any]]): All messages in the conversation.
        tool_calls (List[Dict[str, Any]]): All tool calls made.
        iterations (int): Number of iterations completed.
        start_time (str): ISO timestamp of conversation start.
        end_time (Optional[str]): ISO timestamp of conversation end.
        final_response (Optional[str]): The final response text.
        metadata (Dict[str, Any]): Additional metadata.
    """
    
    def __init__(self, conversation_id: str):
        """
        Initialize conversation state.
        
        Args:
            conversation_id (str): Unique identifier for this conversation.
        """
        self.conversation_id = conversation_id
        self.messages: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.iterations = 0
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        self.final_response: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """
        Add a message to the conversation state.
        
        Args:
            role (str): The role of the message sender.
            content (str): The message content.
            **kwargs: Additional message attributes.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(message)
    
    def add_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Add a tool call to the conversation state.
        
        Args:
            tool_name (str): Name of the tool.
            tool_args (Dict[str, Any]): Arguments passed to the tool.
            result (str): The result of the tool call.
            success (bool): Whether the call succeeded.
            error (Optional[str]): Error message if failed.
        """
        tool_call = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iterations
        }
        self.tool_calls.append(tool_call)
    
    def increment_iteration(self) -> int:
        """
        Increment the iteration counter.
        
        Returns:
            int: The new iteration count.
        """
        self.iterations += 1
        return self.iterations
    
    def finalize(self, final_response: str) -> None:
        """
        Mark the conversation as complete.
        
        Args:
            final_response (str): The final response text.
        """
        self.end_time = datetime.now().isoformat()
        self.final_response = final_response
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the conversation state to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation.
        """
        return {
            "conversation_id": self.conversation_id,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "final_response": self.final_response,
            "metadata": self.metadata,
            "message_count": len(self.messages),
            "tool_call_count": len(self.tool_calls)
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert the conversation state to a JSON string.
        
        Args:
            indent (int): Indentation level for pretty printing.
            
        Returns:
            str: JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the conversation state to a file.
        
        Args:
            filepath (str): Path to save the file.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ConversationState':
        """
        Load a conversation state from a file.
        
        Args:
            filepath (str): Path to the file.
            
        Returns:
            ConversationState: The loaded conversation state.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        state = cls(data["conversation_id"])
        state.messages = data.get("messages", [])
        state.tool_calls = data.get("tool_calls", [])
        state.iterations = data.get("iterations", 0)
        state.start_time = data.get("start_time", "")
        state.end_time = data.get("end_time")
        state.final_response = data.get("final_response")
        state.metadata = data.get("metadata", {})
        
        return state


class MessageHistoryInspector:
    """
    Utility for inspecting and analyzing message history.
    
    This class provides methods for analyzing conversation messages
    to understand agent behavior and debug issues.
    
    Attributes:
        state (ConversationState): The conversation state to inspect.
    """
    
    def __init__(self, state: ConversationState):
        """
        Initialize the inspector.
        
        Args:
            state (ConversationState): The conversation state to inspect.
        """
        self.state = state
    
    def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Get all messages with a specific role.
        
        Args:
            role (str): The role to filter by.
            
        Returns:
            List[Dict[str, Any]]: List of matching messages.
        """
        return [m for m in self.state.messages if m.get("role") == role]
    
    def get_tool_call_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all tool calls.
        
        Returns:
            List[Dict[str, Any]]: List of tool call summaries.
        """
        return [
            {
                "tool_name": tc["tool_name"],
                "success": tc["success"],
                "iteration": tc["iteration"],
                "result_preview": tc["result"][:100] + "..." if len(tc["result"]) > 100 else tc["result"]
            }
            for tc in self.state.tool_calls
        ]
    
    def get_failed_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Get all failed tool calls.
        
        Returns:
            List[Dict[str, Any]]: List of failed tool calls.
        """
        return [tc for tc in self.state.tool_calls if not tc.get("success", True)]
    
    def get_iteration_breakdown(self) -> Dict[int, Dict[str, Any]]:
        """
        Get a breakdown of activity by iteration.
        
        Returns:
            Dict[int, Dict[str, Any]]: Breakdown by iteration number.
        """
        breakdown = {}
        
        for i in range(1, self.state.iterations + 1):
            iteration_messages = [m for m in self.state.messages if m.get("iteration") == i]
            iteration_tools = [tc for tc in self.state.tool_calls if tc.get("iteration") == i]
            
            breakdown[i] = {
                "message_count": len(iteration_messages),
                "tool_call_count": len(iteration_tools),
                "tools_called": [tc["tool_name"] for tc in iteration_tools],
                "successes": sum(1 for tc in iteration_tools if tc.get("success")),
                "failures": sum(1 for tc in iteration_tools if not tc.get("success"))
            }
        
        return breakdown
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of the conversation.
        
        Returns:
            Dict[str, Any]: Analysis results.
        """
        return {
            "conversation_id": self.state.conversation_id,
            "total_messages": len(self.state.messages),
            "total_tool_calls": len(self.state.tool_calls),
            "iterations": self.state.iterations,
            "message_roles": {
                role: len(self.get_messages_by_role(role))
                for role in set(m.get("role") for m in self.state.messages)
            },
            "tool_usage": {
                tool_name: sum(1 for tc in self.state.tool_calls if tc["tool_name"] == tool_name)
                for tool_name in set(tc["tool_name"] for tc in self.state.tool_calls)
            },
            "failed_tool_calls": len(self.get_failed_tool_calls()),
            "iteration_breakdown": self.get_iteration_breakdown(),
            "duration_seconds": self._calculate_duration()
        }
    
    def _calculate_duration(self) -> Optional[float]:
        """
        Calculate the duration of the conversation in seconds.
        
        Returns:
            Optional[float]: Duration in seconds, or None if not complete.
        """
        if not self.state.start_time or not self.state.end_time:
            return None
        
        start = datetime.fromisoformat(self.state.start_time)
        end = datetime.fromisoformat(self.state.end_time)
        return (end - start).total_seconds()


class ConversationReplay:
    """
    Utility for replaying conversations for debugging.
    
    This class provides methods to replay a saved conversation
    step by step to understand agent behavior.
    """
    
    def __init__(self, state: ConversationState):
        """
        Initialize the replay utility.
        
        Args:
            state (ConversationState): The conversation state to replay.
        """
        self.state = state
        self.current_step = 0
        self.total_steps = len(state.messages) + len(state.tool_calls)
    
    def reset(self) -> None:
        """
        Reset the replay to the beginning.
        """
        self.current_step = 0
    
    def get_current_step(self) -> Dict[str, Any]:
        """
        Get the current step in the replay.
        
        Returns:
            Dict[str, Any]: The current step data.
        """
        if self.current_step >= self.total_steps:
            return {"type": "end", "message": "Replay complete"}
        
        # Interleave messages and tool calls by timestamp
        all_events = []
        for m in self.state.messages:
            all_events.append({"type": "message", "data": m, "timestamp": m.get("timestamp", "")})
        for tc in self.state.tool_calls:
            all_events.append({"type": "tool_call", "data": tc, "timestamp": tc.get("timestamp", "")})
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x["timestamp"])
        
        if self.current_step < len(all_events):
            return all_events[self.current_step]
        
        return {"type": "end", "message": "Replay complete"}
    
    def step_forward(self) -> Dict[str, Any]:
        """
        Move to the next step in the replay.
        
        Returns:
            Dict[str, Any]: The next step data.
        """
        step = self.get_current_step()
        self.current_step += 1
        return step
    
    def step_backward(self) -> Dict[str, Any]:
        """
        Move to the previous step in the replay.
        
        Returns:
            Dict[str, Any]: The previous step data.
        """
        if self.current_step > 0:
            self.current_step -= 1
        return self.get_current_step()
    
    def get_progress(self) -> Dict[str, int]:
        """
        Get the current progress of the replay.
        
        Returns:
            Dict[str, int]: Progress information.
        """
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": round(self.current_step / self.total_steps * 100, 1) if self.total_steps > 0 else 0
        }


def export_conversation_for_debugging(
    state: ConversationState,
    output_dir: str = "logs/debug",
    include_full_outputs: bool = True
) -> Dict[str, str]:
    """
    Export a conversation state for debugging purposes.
    
    Creates multiple output files for different debugging views.
    
    Args:
        state (ConversationState): The conversation state to export.
        output_dir (str): Directory for output files.
        include_full_outputs (bool): Include full tool outputs.
        
    Returns:
        Dict[str, str]: Paths to created files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"conversation_{state.conversation_id[:8]}_{timestamp}"
    
    files_created = {}
    
    # Full state export
    full_path = os.path.join(output_dir, f"{base_filename}_full.json")
    state.save_to_file(full_path)
    files_created["full_state"] = full_path
    
    # Analysis export
    inspector = MessageHistoryInspector(state)
    analysis = inspector.analyze()
    analysis_path = os.path.join(output_dir, f"{base_filename}_analysis.json")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, default=str)
    files_created["analysis"] = analysis_path
    
    # Human-readable summary
    summary_path = os.path.join(output_dir, f"{base_filename}_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Conversation Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Conversation ID: {state.conversation_id}\n")
        f.write(f"Start Time: {state.start_time}\n")
        f.write(f"End Time: {state.end_time or 'N/A'}\n")
        f.write(f"Iterations: {state.iterations}\n")
        f.write(f"Total Messages: {len(state.messages)}\n")
        f.write(f"Total Tool Calls: {len(state.tool_calls)}\n\n")
        
        f.write(f"Tool Calls:\n")
        f.write(f"{'-'*30}\n")
        for tc in state.tool_calls:
            status = "SUCCESS" if tc.get("success") else "FAILED"
            f.write(f"  [{status}] {tc['tool_name']}\n")
            if include_full_outputs:
                f.write(f"    Args: {json.dumps(tc['tool_args'], default=str)}\n")
                result_preview = tc['result'][:200] + "..." if len(tc['result']) > 200 else tc['result']
                f.write(f"    Result: {result_preview}\n")
        
        f.write(f"\nFinal Response:\n")
        f.write(f"{'-'*30}\n")
        f.write(f"{state.final_response or 'N/A'}\n")
    
    files_created["summary"] = summary_path
    
    return files_created


def create_debug_context(
    conversation_id: str,
    debug_mode: Optional[DebugMode] = None
) -> Dict[str, Any]:
    """
    Create a debug context for a conversation.
    
    This function creates a complete debug context including
    conversation state and debug mode configuration.
    
    Args:
        conversation_id (str): Unique identifier for the conversation.
        debug_mode (Optional[DebugMode]): Debug mode configuration.
        
    Returns:
        Dict[str, Any]: Debug context dictionary.
    """
    return {
        "conversation_state": ConversationState(conversation_id),
        "debug_mode": debug_mode or DebugMode(),
        "inspector": None  # Will be set after conversation completes
    }