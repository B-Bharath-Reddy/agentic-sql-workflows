"""
Tests for the debug_utils module.

This module tests the debugging utilities including debug mode configuration,
conversation state management, and message history inspection.
"""

import pytest
import json
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.debug_utils import (
    DebugMode, ConversationState, MessageHistoryInspector,
    ConversationReplay, export_conversation_for_debugging,
    create_debug_context
)


class TestDebugMode:
    """Tests for the DebugMode class."""
    
    def test_debug_mode_defaults(self):
        """Test default debug mode settings."""
        debug = DebugMode()
        
        assert debug.enabled is False
        assert debug.verbose is False
        assert debug.capture_messages is True
        assert debug.capture_tool_outputs is True
        assert debug.output_dir == "logs/debug"
    
    def test_debug_mode_enabled(self):
        """Test enabled debug mode."""
        debug = DebugMode(enabled=True)
        
        assert debug.enabled is True
        assert debug.should_capture() is True
    
    def test_debug_mode_verbose(self):
        """Test verbose mode."""
        debug = DebugMode(enabled=True, verbose=True)
        
        assert debug.should_print_verbose() is True
    
    def test_should_capture(self):
        """Test should_capture method."""
        debug_disabled = DebugMode(enabled=False)
        debug_enabled = DebugMode(enabled=True)
        
        assert debug_disabled.should_capture() is False
        assert debug_enabled.should_capture() is True
    
    def test_custom_output_dir(self):
        """Test custom output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            debug = DebugMode(enabled=True, output_dir=tmpdir)
            
            assert debug.output_dir == tmpdir
            # Directory should be created when enabled
            assert os.path.exists(tmpdir)
    
    def test_save_debug_file(self):
        """Test saving debug files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            debug = DebugMode(enabled=True, output_dir=tmpdir)
            
            filepath = debug.save_debug_file("test.json", {"test": "data"})
            
            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert data["test"] == "data"
    
    def test_save_debug_file_disabled(self):
        """Test that save_debug_file does nothing when disabled."""
        debug = DebugMode(enabled=False)
        
        filepath = debug.save_debug_file("test.json", {"test": "data"})
        
        assert filepath == ""


class TestConversationState:
    """Tests for the ConversationState class."""
    
    def test_state_creation(self):
        """Test creating conversation state."""
        state = ConversationState(conversation_id="conv-123")
        
        assert state.conversation_id == "conv-123"
        assert state.messages == []
        assert state.tool_calls == []
        assert state.iterations == 0
        assert state.start_time is not None
        assert state.end_time is None
        assert state.final_response is None
    
    def test_add_message(self):
        """Test adding messages."""
        state = ConversationState(conversation_id="conv-123")
        
        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi there!")
        
        assert len(state.messages) == 2
        assert state.messages[0]["role"] == "user"
        assert state.messages[0]["content"] == "Hello"
        assert state.messages[1]["role"] == "assistant"
    
    def test_add_message_with_iteration(self):
        """Test adding message with iteration info."""
        state = ConversationState(conversation_id="conv-123")
        
        state.add_message("user", "Hello", iteration=1)
        
        assert state.messages[0]["iteration"] == 1
    
    def test_add_message_with_tool_calls(self):
        """Test adding message with tool call info."""
        state = ConversationState(conversation_id="conv-123")
        
        state.add_message("assistant", "Let me check", tool_calls=[{"name": "query"}])
        
        assert state.messages[0]["tool_calls"] == [{"name": "query"}]
    
    def test_add_tool_call(self):
        """Test adding tool calls."""
        state = ConversationState(conversation_id="conv-123")
        
        state.add_tool_call(
            tool_name="execute_sql",
            tool_args={"query": "SELECT 1"},
            result="result",
            success=True
        )
        
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0]["tool_name"] == "execute_sql"
        assert state.tool_calls[0]["success"] is True
    
    def test_add_tool_call_failure(self):
        """Test adding failed tool call."""
        state = ConversationState(conversation_id="conv-123")
        
        state.add_tool_call(
            tool_name="execute_sql",
            tool_args={"query": "INVALID"},
            result="",
            success=False,
            error="Syntax error"
        )
        
        assert state.tool_calls[0]["success"] is False
        assert state.tool_calls[0]["error"] == "Syntax error"
    
    def test_increment_iteration(self):
        """Test incrementing iterations."""
        state = ConversationState(conversation_id="conv-123")
        
        assert state.increment_iteration() == 1
        assert state.iterations == 1
        assert state.increment_iteration() == 2
        assert state.iterations == 2
    
    def test_finalize(self):
        """Test finalizing conversation."""
        state = ConversationState(conversation_id="conv-123")
        
        state.finalize("Final answer")
        
        assert state.end_time is not None
        assert state.final_response == "Final answer"
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message("user", "Hello")
        state.add_tool_call("tool", {}, "result")
        state.increment_iteration()
        state.finalize("Done")
        
        result = state.to_dict()
        
        assert result["conversation_id"] == "conv-123"
        assert result["message_count"] == 1
        assert result["tool_call_count"] == 1
        assert result["iterations"] == 1
        assert result["final_response"] == "Done"
    
    def test_to_json(self):
        """Test JSON serialization."""
        state = ConversationState(conversation_id="conv-123")
        
        json_str = state.to_json()
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert data["conversation_id"] == "conv-123"
    
    def test_save_and_load_file(self):
        """Test saving and loading state from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ConversationState(conversation_id="conv-123")
            state.add_message("user", "Hello")
            state.add_tool_call("tool", {"arg": "value"}, "result")
            state.increment_iteration()
            state.finalize("Done")
            
            filepath = os.path.join(tmpdir, "state.json")
            state.save_to_file(filepath)
            
            # Load the state
            loaded = ConversationState.load_from_file(filepath)
            
            assert loaded.conversation_id == "conv-123"
            assert len(loaded.messages) == 1
            assert len(loaded.tool_calls) == 1
            assert loaded.iterations == 1
            assert loaded.final_response == "Done"


class TestMessageHistoryInspector:
    """Tests for the MessageHistoryInspector class."""
    
    def test_inspector_creation(self):
        """Test creating an inspector."""
        state = ConversationState(conversation_id="conv-123")
        inspector = MessageHistoryInspector(state)
        
        assert inspector.state is state
    
    def test_get_messages_by_role(self):
        """Test filtering messages by role."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi")
        state.add_message("user", "How are you?")
        
        inspector = MessageHistoryInspector(state)
        user_messages = inspector.get_messages_by_role("user")
        
        assert len(user_messages) == 2
        assert all(m["role"] == "user" for m in user_messages)
    
    def test_get_tool_call_summary(self):
        """Test getting tool call summary."""
        state = ConversationState(conversation_id="conv-123")
        state.add_tool_call("tool1", {}, "short result", success=True)
        state.add_tool_call("tool2", {}, "x" * 200, success=False)
        
        inspector = MessageHistoryInspector(state)
        summary = inspector.get_tool_call_summary()
        
        assert len(summary) == 2
        assert summary[0]["tool_name"] == "tool1"
        assert summary[0]["success"] is True
        assert summary[1]["success"] is False
        # Long result should be truncated
        assert "..." in summary[1]["result_preview"]
    
    def test_get_failed_tool_calls(self):
        """Test getting failed tool calls."""
        state = ConversationState(conversation_id="conv-123")
        state.add_tool_call("tool1", {}, "result", success=True)
        state.add_tool_call("tool2", {}, "", success=False)
        state.add_tool_call("tool3", {}, "", success=False)
        
        inspector = MessageHistoryInspector(state)
        failed = inspector.get_failed_tool_calls()
        
        assert len(failed) == 2
    
    def test_get_iteration_breakdown(self):
        """Test getting iteration breakdown."""
        state = ConversationState(conversation_id="conv-123")
        state.increment_iteration()
        state.add_tool_call("tool1", {}, "result", success=True)
        state.increment_iteration()
        state.add_tool_call("tool2", {}, "", success=False)
        
        inspector = MessageHistoryInspector(state)
        breakdown = inspector.get_iteration_breakdown()
        
        assert 1 in breakdown
        assert 2 in breakdown
        assert breakdown[1]["tool_call_count"] == 1
        assert breakdown[2]["failures"] == 1
    
    def test_analyze(self):
        """Test comprehensive analysis."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi")
        state.add_tool_call("tool1", {}, "result")
        state.increment_iteration()
        state.finalize("Done")
        
        inspector = MessageHistoryInspector(state)
        analysis = inspector.analyze()
        
        assert analysis["conversation_id"] == "conv-123"
        assert analysis["total_messages"] == 2
        assert analysis["total_tool_calls"] == 1
        assert "message_roles" in analysis
        assert "tool_usage" in analysis


class TestConversationReplay:
    """Tests for the ConversationReplay class."""
    
    def test_replay_creation(self):
        """Test creating a replay."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message("user", "Hello")
        state.add_tool_call("tool", {}, "result")
        
        replay = ConversationReplay(state)
        
        assert replay.current_step == 0
        assert replay.total_steps == 2
    
    def test_step_forward(self):
        """Test stepping forward in replay."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message("user", "Hello")
        state.add_tool_call("tool", {}, "result")
        
        replay = ConversationReplay(state)
        
        step1 = replay.step_forward()
        assert replay.current_step == 1
        assert step1["type"] in ["message", "tool_call"]
    
    def test_step_backward(self):
        """Test stepping backward in replay."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message("user", "Hello")
        
        replay = ConversationReplay(state)
        replay.step_forward()
        
        step = replay.step_backward()
        assert replay.current_step == 0
    
    def test_reset(self):
        """Test resetting replay."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message("user", "Hello")
        
        replay = ConversationReplay(state)
        replay.step_forward()
        replay.reset()
        
        assert replay.current_step == 0
    
    def test_get_progress(self):
        """Test getting progress."""
        state = ConversationState(conversation_id="conv-123")
        state.add_message("user", "Hello")
        state.add_tool_call("tool", {}, "result")
        
        replay = ConversationReplay(state)
        replay.step_forward()
        
        progress = replay.get_progress()
        
        assert progress["current_step"] == 1
        assert progress["total_steps"] == 2
        assert progress["progress_percent"] == 50.0


class TestExportConversationForDebugging:
    """Tests for the export_conversation_for_debugging function."""
    
    def test_export_creates_directory(self):
        """Test that export creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "debug_output")
            state = ConversationState(conversation_id="conv-123")
            
            files = export_conversation_for_debugging(state, output_dir)
            
            assert os.path.exists(output_dir)
    
    def test_export_creates_files(self):
        """Test that export creates expected files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ConversationState(conversation_id="conv-123")
            state.add_message("user", "Hello")
            state.finalize("Done")
            
            files = export_conversation_for_debugging(state, tmpdir)
            
            assert "full_state" in files
            assert "analysis" in files
            assert "summary" in files
            assert os.path.exists(files["full_state"])
            assert os.path.exists(files["analysis"])
            assert os.path.exists(files["summary"])
    
    def test_export_with_empty_state(self):
        """Test exporting empty state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ConversationState(conversation_id="conv-123")
            
            files = export_conversation_for_debugging(state, tmpdir)
            
            # Should still create files
            assert os.path.exists(files["full_state"])


class TestCreateDebugContext:
    """Tests for the create_debug_context function."""
    
    def test_create_debug_context(self):
        """Test creating a debug context."""
        context = create_debug_context("conv-123")
        
        assert "conversation_state" in context
        assert "debug_mode" in context
        assert isinstance(context["conversation_state"], ConversationState)
        assert isinstance(context["debug_mode"], DebugMode)
    
    def test_create_debug_context_with_debug_mode(self):
        """Test creating debug context with custom debug mode."""
        debug = DebugMode(enabled=True, verbose=True)
        context = create_debug_context("conv-123", debug_mode=debug)
        
        assert context["debug_mode"].enabled is True
        assert context["debug_mode"].verbose is True