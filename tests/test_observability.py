"""
Tests for the observability module.

This module tests the unified observability layer including context management,
metrics integration, and tracing coordination.
"""

import pytest
import json
import time
import logging
from unittest.mock import patch, MagicMock

from src.observability import (
    ObservabilityContext, ObservabilityManager,
    get_observability_manager, create_observability_context,
    instrument_tool, instrument_llm_call
)
from src.metrics import reset_metrics_collector


class TestObservabilityContext:
    """Tests for the ObservabilityContext class."""
    
    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics_collector()
    
    def test_context_creation(self):
        """Test creating an observability context."""
        context = ObservabilityContext(
            user_query="What is the product count?"
        )
        
        assert context.trace_id is not None
        assert context.query_metrics is not None
        assert context.start_time is not None
    
    def test_context_with_metadata(self):
        """Test creating context with metadata."""
        context = ObservabilityContext(
            user_query="Test query",
            metadata={"user_id": "user-123", "session": "session-456"}
        )
        
        assert context.trace_id is not None
    
    def test_start_iteration(self):
        """Test starting iterations."""
        context = ObservabilityContext(user_query="test")
        
        iter1 = context.start_iteration()
        iter2 = context.start_iteration()
        
        assert iter1 == 1
        assert iter2 == 2
    
    def test_record_llm_call(self):
        """Test recording an LLM call."""
        context = ObservabilityContext(user_query="test")
        
        result = context.record_llm_call(
            model="llama-3.3-70b-versatile",
            input_text="What is 2+2?",
            output_text="The answer is 4",
            latency_ms=500.0
        )
        
        assert "model" in result
        assert result["model"] == "llama-3.3-70b-versatile"
        assert len(context.query_metrics.llm_calls) == 1
    
    def test_record_llm_call_failure(self):
        """Test recording a failed LLM call."""
        context = ObservabilityContext(user_query="test")
        
        result = context.record_llm_call(
            model="llama-3.3-70b-versatile",
            input_text="What is 2+2?",
            output_text="",
            latency_ms=100.0,
            success=False,
            error_message="Rate limit exceeded"
        )
        
        assert result["success"] is False
        assert result["error_message"] == "Rate limit exceeded"
    
    def test_record_tool_call(self):
        """Test recording a tool call."""
        context = ObservabilityContext(user_query="test")
        
        result = context.record_tool_call(
            tool_name="execute_sql_query",
            tool_args={"query": "SELECT COUNT(*) FROM products"},
            result="100 rows",
            latency_ms=150.0
        )
        
        assert result["tool_name"] == "execute_sql_query"
        assert result["success"] is True
        assert len(context.query_metrics.tool_calls) == 1
    
    def test_record_tool_call_failure(self):
        """Test recording a failed tool call."""
        context = ObservabilityContext(user_query="test")
        
        result = context.record_tool_call(
            tool_name="execute_sql_query",
            tool_args={"query": "INVALID SQL"},
            result="",
            latency_ms=50.0,
            success=False,
            error_message="Syntax error near 'INVALID'"
        )
        
        assert result["success"] is False
        assert result["error_message"] == "Syntax error near 'INVALID'"
    
    def test_finalize_success(self):
        """Test finalizing a successful context."""
        context = ObservabilityContext(user_query="test")
        context.start_iteration()
        
        summary = context.finalize(success=True, final_response="Done")
        
        assert summary["finalized"] is True
        assert context._finalized is True
    
    def test_finalize_failure(self):
        """Test finalizing a failed context."""
        context = ObservabilityContext(user_query="test")
        
        summary = context.finalize(success=False)
        
        assert summary["finalized"] is True
    
    def test_to_dict(self):
        """Test exporting debug state."""
        context = ObservabilityContext(
            user_query="test",
            metadata={"key": "value"}
        )
        context.start_iteration()
        
        debug_state = context.export_debug_state()
        
        assert "trace" in debug_state
        assert "metrics" in debug_state
        assert "summary" in debug_state
    
    def test_get_summary(self):
        """Test getting summary."""
        context = ObservabilityContext(user_query="test")
        context.record_llm_call("model", "input", "output", 100.0)
        context.record_tool_call("tool", {}, "result", 50.0)
        context.start_iteration()
        
        summary = context._get_summary()
        
        assert summary["trace_id"] == context.trace_id
        assert summary["iterations"] == 1
        assert summary["llm_calls"] == 1
        assert summary["tool_calls"] == 1


class TestObservabilityManager:
    """Tests for the ObservabilityManager class."""
    
    def setup_method(self):
        """Reset before each test."""
        reset_metrics_collector()
        # Reset the global manager
        import src.observability as obs_module
        obs_module._observability_manager = None
    
    def test_manager_creation(self):
        """Test creating a manager."""
        manager = ObservabilityManager()
        
        assert manager.contexts == {}
        assert manager.debug_mode is False
    
    def test_create_context(self):
        """Test creating a context through manager."""
        manager = ObservabilityManager()
        
        context = manager.create_context(user_query="What is the product count?")
        
        assert context.trace_id in manager.contexts
        assert manager.contexts[context.trace_id] is context
    
    def test_get_context(self):
        """Test getting an existing context."""
        manager = ObservabilityManager()
        created = manager.create_context(user_query="test")
        
        retrieved = manager.get_context(created.trace_id)
        
        assert retrieved is created
    
    def test_get_context_nonexistent(self):
        """Test getting a nonexistent context."""
        manager = ObservabilityManager()
        
        result = manager.get_context("nonexistent-id")
        
        assert result is None
    
    def test_remove_context(self):
        """Test removing a context."""
        manager = ObservabilityManager()
        context = manager.create_context(user_query="test")
        
        manager.remove_context(context.trace_id)
        
        assert context.trace_id not in manager.contexts
    
    def test_get_all_active_contexts(self):
        """Test getting all active contexts."""
        manager = ObservabilityManager()
        c1 = manager.create_context(user_query="query 1")
        c2 = manager.create_context(user_query="query 2")
        
        all_contexts = manager.get_all_active_contexts()
        
        assert len(all_contexts) == 2
        assert c1.trace_id in all_contexts
        assert c2.trace_id in all_contexts
    
    def test_get_global_metrics_summary(self):
        """Test getting global metrics summary."""
        manager = ObservabilityManager()
        
        summary = manager.get_global_metrics_summary()
        
        assert "total_queries" in summary
        assert "total_tokens_used" in summary


class TestObservabilityManagerSingleton:
    """Tests for the global observability manager singleton."""
    
    def setup_method(self):
        """Reset before each test."""
        import src.observability as obs_module
        obs_module._observability_manager = None
    
    def test_get_manager_returns_instance(self):
        """Test that get_manager returns an instance."""
        manager = get_observability_manager()
        
        assert manager is not None
        assert isinstance(manager, ObservabilityManager)
    
    def test_get_manager_with_logger(self):
        """Test getting manager with custom logger."""
        logger = logging.getLogger("test_logger")
        manager = get_observability_manager(logger=logger)
        
        assert manager.logger is logger
    
    def test_get_manager_with_debug_mode(self):
        """Test getting manager with debug mode enabled."""
        manager = get_observability_manager(debug_mode=True)
        
        assert manager.debug_mode is True


class TestCreateObservabilityContext:
    """Tests for the create_observability_context convenience function."""
    
    def setup_method(self):
        """Reset before each test."""
        import src.observability as obs_module
        obs_module._observability_manager = None
        reset_metrics_collector()
    
    def test_create_context(self):
        """Test creating a context via convenience function."""
        context = create_observability_context(user_query="test query")
        
        assert context is not None
        assert context.trace_id is not None


class TestObservabilityIntegration:
    """Integration tests for the observability system."""
    
    def setup_method(self):
        """Reset before each test."""
        import src.observability as obs_module
        obs_module._observability_manager = None
        reset_metrics_collector()
    
    def test_full_query_lifecycle(self):
        """Test a complete query lifecycle with observability."""
        manager = ObservabilityManager()
        
        # Create context
        context = manager.create_context(
            user_query="How many products do we have?",
            metadata={"source": "api"}
        )
        
        # Simulate agent iterations
        context.start_iteration()
        context.record_llm_call(
            model="llama-3.3-70b-versatile",
            input_text="User asks: How many products?",
            output_text="I need to query the database",
            latency_ms=300.0
        )
        context.record_tool_call(
            tool_name="execute_sql_query",
            tool_args={"query": "SELECT COUNT(*) FROM products"},
            result="150",
            latency_ms=100.0
        )
        
        context.start_iteration()
        context.record_llm_call(
            model="llama-3.3-70b-versatile",
            input_text="The result is 150",
            output_text="There are 150 products",
            latency_ms=200.0
        )
        
        # Finalize
        summary = context.finalize(
            success=True,
            final_response="There are 150 products in the database."
        )
        
        assert summary["iterations"] == 2
        assert summary["llm_calls"] == 2
        assert summary["tool_calls"] == 1
        assert summary["finalized"] is True
        
        # Check global metrics
        metrics_summary = manager.get_global_metrics_summary()
        assert metrics_summary["total_queries"] == 1
        assert metrics_summary["successful_queries"] == 1
    
    def test_error_handling_lifecycle(self):
        """Test error handling in observability."""
        manager = ObservabilityManager()
        
        context = manager.create_context(user_query="test")
        
        context.start_iteration()
        context.record_tool_call(
            tool_name="execute_sql_query",
            tool_args={},
            result="",
            latency_ms=50.0,
            success=False,
            error_message="Connection failed"
        )
        
        summary = context.finalize(success=False)
        
        assert summary["finalized"] is True
        
        metrics_summary = manager.get_global_metrics_summary()
        assert metrics_summary["failed_queries"] == 1


class TestObservabilityExport:
    """Tests for observability export functionality."""
    
    def setup_method(self):
        """Reset before each test."""
        import src.observability as obs_module
        obs_module._observability_manager = None
        reset_metrics_collector()
    
    def test_context_to_json(self):
        """Test exporting context to JSON."""
        context = ObservabilityContext(
            user_query="test query",
            metadata={"key": "value"}
        )
        
        debug_state = context.export_debug_state()
        
        # Should be JSON serializable
        json_str = json.dumps(debug_state, default=str)
        assert json_str is not None
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert "summary" in parsed
    
    def test_manager_export_all(self):
        """Test exporting all contexts from manager."""
        manager = ObservabilityManager()
        
        c1 = manager.create_context(user_query="query 1")
        c2 = manager.create_context(user_query="query 2")
        
        all_contexts = manager.get_all_active_contexts()
        
        assert len(all_contexts) == 2


class TestInstrumentTool:
    """Tests for the instrument_tool decorator."""
    
    def setup_method(self):
        """Reset before each test."""
        reset_metrics_collector()
    
    def test_instrument_tool_success(self):
        """Test instrumenting a successful tool call."""
        @instrument_tool
        def my_tool(arg1, arg2):
            return "result"
        
        result = my_tool("a", "b")
        
        assert result == "result"
    
    def test_instrument_tool_failure(self):
        """Test instrumenting a failing tool call."""
        @instrument_tool
        def failing_tool():
            raise ValueError("Tool failed")
        
        with pytest.raises(ValueError, match="Tool failed"):
            failing_tool()


class TestInstrumentLLMCall:
    """Tests for the instrument_llm_call decorator."""
    
    def setup_method(self):
        """Reset before each test."""
        reset_metrics_collector()
    
    def test_instrument_llm_success(self):
        """Test instrumenting a successful LLM call."""
        @instrument_llm_call
        def call_llm(prompt):
            return "response"
        
        result = call_llm("test prompt")
        
        assert result == "response"
    
    def test_instrument_llm_failure(self):
        """Test instrumenting a failing LLM call."""
        @instrument_llm_call
        def failing_llm():
            raise RuntimeError("API error")
        
        with pytest.raises(RuntimeError, match="API error"):
            failing_llm()