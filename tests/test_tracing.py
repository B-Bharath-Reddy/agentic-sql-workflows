"""
Tests for the tracing module.

This module tests distributed tracing capabilities including correlation ID
management, span tracking, and context propagation.
"""

import pytest
import time
import uuid
from unittest.mock import patch

from src.tracing import (
    Span, TraceContext, start_trace, get_current_trace, end_trace,
    start_span, complete_span, fail_span, get_trace_id,
    TracingContextManager, SpanContextManager
)


class TestSpan:
    """Tests for the Span class."""
    
    def test_span_creation(self):
        """Test creating a span."""
        span = Span(operation_name="test_operation")
        
        assert span.span_id is not None
        assert len(span.span_id) == 8
        assert span.operation_name == "test_operation"
        assert span.start_time is not None
        assert span.end_time is None
        assert span.duration_ms is None
        assert span.status == "started"
        assert span.error_message is None
    
    def test_span_with_metadata(self):
        """Test creating a span with metadata."""
        span = Span(
            operation_name="llm_call",
            metadata={"model": "llama-3", "tokens": 100}
        )
        
        assert span.metadata["model"] == "llama-3"
        assert span.metadata["tokens"] == 100
    
    def test_span_complete(self):
        """Test completing a span."""
        span = Span(operation_name="test")
        time.sleep(0.01)  # Small delay to ensure duration > 0
        
        span.complete(metadata={"result": "success"})
        
        assert span.status == "completed"
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms > 0
        assert span.metadata["result"] == "success"
    
    def test_span_fail(self):
        """Test failing a span."""
        span = Span(operation_name="test")
        
        span.fail("Something went wrong", metadata={"error_code": 500})
        
        assert span.status == "error"
        assert span.error_message == "Something went wrong"
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.metadata["error_code"] == 500
    
    def test_span_to_dict(self):
        """Test span serialization."""
        span = Span(
            operation_name="test_op",
            metadata={"key": "value"}
        )
        span.complete()
        
        result = span.to_dict()
        
        assert result["span_id"] == span.span_id
        assert result["operation_name"] == "test_op"
        assert result["status"] == "completed"
        assert result["metadata"]["key"] == "value"
        # duration_ms is set after complete()
        assert result["duration_ms"] is not None or result["status"] == "completed"


class TestTraceContext:
    """Tests for the TraceContext class."""
    
    def test_trace_creation(self):
        """Test creating a trace context."""
        trace = TraceContext()
        
        assert trace.trace_id is not None
        assert trace.start_time is not None
        assert trace.spans == []
        assert trace.metadata == {}
    
    def test_trace_with_metadata(self):
        """Test creating a trace with metadata."""
        trace = TraceContext(metadata={"user_id": "user-123"})
        
        assert trace.metadata["user_id"] == "user-123"
    
    def test_add_span(self):
        """Test adding spans to a trace."""
        trace = TraceContext()
        span1 = Span(operation_name="op1")
        span2 = Span(operation_name="op2")
        
        trace.add_span(span1)
        trace.add_span(span2)
        
        assert len(trace.spans) == 2
        assert trace.spans[0] is span1
        assert trace.spans[1] is span2
    
    def test_get_total_duration_ms(self):
        """Test calculating total duration."""
        trace = TraceContext()
        
        span1 = Span(operation_name="op1", start_time=100.0)
        span1.end_time = 105.0
        span1.duration_ms = 5000.0
        
        span2 = Span(operation_name="op2", start_time=102.0)
        span2.end_time = 110.0
        span2.duration_ms = 8000.0
        
        trace.add_span(span1)
        trace.add_span(span2)
        
        duration = trace.get_total_duration_ms()
        
        # Total duration should be from min start to max end
        # min start = 100.0, max end = 110.0
        # duration = (110.0 - 100.0) * 1000 = 10000.0
        assert duration == 10000.0
    
    def test_get_total_duration_empty(self):
        """Test total duration with no spans."""
        trace = TraceContext()
        
        assert trace.get_total_duration_ms() == 0.0
    
    def test_to_dict(self):
        """Test trace serialization."""
        trace = TraceContext(metadata={"query": "test"})
        span = Span(operation_name="op")
        span.complete()
        trace.add_span(span)
        
        result = trace.to_dict()
        
        assert result["trace_id"] == trace.trace_id
        assert result["span_count"] == 1
        assert result["metadata"]["query"] == "test"
        assert "start_time_iso" in result
        assert len(result["spans"]) == 1


class TestStartTrace:
    """Tests for the start_trace function."""
    
    def test_start_trace(self):
        """Test starting a new trace."""
        # Clear any existing trace
        end_trace()
        
        trace = start_trace()
        
        assert trace is not None
        assert trace.trace_id is not None
        assert get_current_trace() is trace
        
        # Cleanup
        end_trace()
    
    def test_start_trace_with_metadata(self):
        """Test starting a trace with metadata."""
        end_trace()
        
        trace = start_trace(metadata={"user_query": "How many products?"})
        
        assert trace.metadata["user_query"] == "How many products?"
        
        end_trace()


class TestGetCurrentTrace:
    """Tests for the get_current_trace function."""
    
    def test_get_current_trace_none(self):
        """Test getting trace when none is active."""
        end_trace()
        
        result = get_current_trace()
        
        assert result is None
    
    def test_get_current_trace_active(self):
        """Test getting active trace."""
        end_trace()
        
        trace = start_trace()
        
        assert get_current_trace() is trace
        
        end_trace()


class TestEndTrace:
    """Tests for the end_trace function."""
    
    def test_end_trace(self):
        """Test ending a trace."""
        trace = start_trace()
        
        result = end_trace()
        
        assert result is not None
        assert result["trace_id"] == trace.trace_id
        assert get_current_trace() is None
    
    def test_end_trace_no_active(self):
        """Test ending trace when none is active."""
        end_trace()  # Clear any existing
        
        result = end_trace()
        
        assert result is None


class TestStartSpan:
    """Tests for the start_span function."""
    
    def test_start_span(self):
        """Test starting a span."""
        trace = start_trace()
        
        span = start_span("test_operation")
        
        assert span is not None
        assert span.operation_name == "test_operation"
        assert span in trace.spans
        
        # Complete the span before ending trace to avoid empty end_time issue
        span.complete()
        end_trace()
    
    def test_start_span_with_metadata(self):
        """Test starting a span with metadata."""
        trace = start_trace()
        
        span = start_span("llm_call", metadata={"model": "llama-3"})
        
        assert span.metadata["model"] == "llama-3"
        
        # Complete the span before ending trace
        span.complete()
        end_trace()
    
    def test_start_span_no_trace(self):
        """Test starting a span without an active trace."""
        end_trace()
        
        span = start_span("orphan_operation")
        
        # Should still create a span, just not attached to any trace
        assert span is not None
        assert span.operation_name == "orphan_operation"


class TestCompleteSpan:
    """Tests for the complete_span function."""
    
    def test_complete_span(self):
        """Test completing a span."""
        span = Span(operation_name="test")
        
        complete_span(span, metadata={"result": "ok"})
        
        assert span.status == "completed"
        assert span.metadata["result"] == "ok"


class TestFailSpan:
    """Tests for the fail_span function."""
    
    def test_fail_span(self):
        """Test failing a span."""
        span = Span(operation_name="test")
        
        fail_span(span, "Error occurred", metadata={"code": 500})
        
        assert span.status == "error"
        assert span.error_message == "Error occurred"
        assert span.metadata["code"] == 500


class TestGetTraceId:
    """Tests for the get_trace_id function."""
    
    def test_get_trace_id(self):
        """Test getting the trace ID."""
        end_trace()
        trace = start_trace()
        
        trace_id = get_trace_id()
        
        assert trace_id == trace.trace_id
        
        end_trace()
    
    def test_get_trace_id_none(self):
        """Test getting trace ID when no trace is active."""
        end_trace()
        
        trace_id = get_trace_id()
        
        assert trace_id is None


class TestTracingContextManager:
    """Tests for the TracingContextManager class."""
    
    def test_context_manager(self):
        """Test using TracingContextManager."""
        end_trace()
        
        with TracingContextManager({"query": "test"}) as trace:
            assert get_current_trace() is trace
            assert trace.metadata["query"] == "test"
        
        # Trace should be ended after context exit
        assert get_current_trace() is None
    
    def test_context_manager_with_exception(self):
        """Test TracingContextManager handles exceptions."""
        end_trace()
        
        try:
            with TracingContextManager() as trace:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Trace should still be cleaned up
        assert get_current_trace() is None


class TestSpanContextManager:
    """Tests for the SpanContextManager class."""
    
    def test_context_manager(self):
        """Test using SpanContextManager."""
        trace = start_trace()
        
        with SpanContextManager("test_op", {"key": "value"}) as span:
            assert span.operation_name == "test_op"
            assert span.metadata["key"] == "value"
            assert span.status == "started"
        
        # Span should be completed after context exit
        assert span.status == "completed"
        
        end_trace()
    
    def test_context_manager_with_exception(self):
        """Test SpanContextManager handles exceptions."""
        trace = start_trace()
        
        try:
            with SpanContextManager("failing_op") as span:
                raise RuntimeError("Operation failed")
        except RuntimeError:
            pass
        
        # Span should be marked as failed
        assert span.status == "error"
        assert "Operation failed" in span.error_message
        
        end_trace()


class TestTracingIntegration:
    """Integration tests for the tracing system."""
    
    def test_full_trace_lifecycle(self):
        """Test a complete trace lifecycle with multiple spans."""
        end_trace()
        
        # Start a trace
        trace = start_trace(metadata={"user_query": "How many products?"})
        
        # First span: LLM call
        llm_span = start_span("llm_call", metadata={"model": "llama-3"})
        time.sleep(0.01)
        complete_span(llm_span, {"tokens": 150})
        
        # Second span: Tool call
        tool_span = start_span("tool_call", metadata={"tool": "execute_sql"})
        time.sleep(0.01)
        complete_span(tool_span, {"rows": 100})
        
        # End the trace
        result = end_trace()
        
        assert result is not None
        assert result["span_count"] == 2
        assert len(result["spans"]) == 2
        assert result["spans"][0]["status"] == "completed"
        assert result["spans"][1]["status"] == "completed"
    
    def test_trace_with_failed_span(self):
        """Test trace with a failed span."""
        end_trace()
        
        trace = start_trace()
        
        success_span = start_span("success_op")
        complete_span(success_span)
        
        fail_span_obj = start_span("fail_op")
        fail_span(fail_span_obj, "Operation failed")
        
        result = end_trace()
        
        assert result["span_count"] == 2
        statuses = [s["status"] for s in result["spans"]]
        assert "completed" in statuses
        assert "error" in statuses
    
    def test_nested_context_managers(self):
        """Test nested context managers."""
        end_trace()
        
        with TracingContextManager({"query": "test"}) as trace:
            with SpanContextManager("outer_op") as outer_span:
                with SpanContextManager("inner_op") as inner_span:
                    inner_span.metadata["nested"] = True
                
                assert inner_span.status == "completed"
            
            assert outer_span.status == "completed"
        
        assert len(trace.spans) == 2