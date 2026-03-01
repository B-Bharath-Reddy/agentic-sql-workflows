"""
tracing.py

This module provides distributed tracing capabilities for the Agentic Workflow application.
It implements correlation ID management and span tracking to enable end-to-end request
tracing across the agent execution lifecycle.

The tracing system allows each user query to be uniquely identified and tracked through
all operations including LLM calls, tool executions, and error handling.
"""

import uuid
import time
from typing import Optional, Dict, Any, List
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime


# Context variable to store the current trace context
# This enables trace ID propagation across function calls without explicit passing
_current_trace: ContextVar[Optional['TraceContext']] = ContextVar('current_trace', default=None)


@dataclass
class Span:
    """
    Represents a single operation within a trace.
    
    A span captures timing information and metadata for a specific operation
    such as an LLM call, tool execution, or database query.
    
    Attributes:
        span_id (str): Unique identifier for this span.
        operation_name (str): Name of the operation being traced.
        start_time (float): Unix timestamp when the span started.
        end_time (Optional[float]): Unix timestamp when the span ended.
        duration_ms (Optional[float]): Duration of the span in milliseconds.
        metadata (Dict[str, Any]): Additional metadata about the operation.
        status (str): Status of the span ('started', 'completed', 'error').
        error_message (Optional[str]): Error message if the span failed.
    """
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "started"
    error_message: Optional[str] = None
    
    def complete(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the span as completed and calculate duration.
        
        Args:
            metadata (Optional[Dict[str, Any]]): Additional metadata to merge.
        """
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = "completed"
        if metadata:
            self.metadata.update(metadata)
    
    def fail(self, error_message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the span as failed with an error message.
        
        Args:
            error_message (str): Description of the error.
            metadata (Optional[Dict[str, Any]]): Additional metadata to merge.
        """
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = "error"
        self.error_message = error_message
        if metadata:
            self.metadata.update(metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the span to a dictionary for logging and serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the span.
        """
        return {
            "span_id": self.span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms else None,
            "metadata": self.metadata,
            "status": self.status,
            "error_message": self.error_message
        }


@dataclass
class TraceContext:
    """
    Represents the complete trace context for a single request.
    
    A trace context contains the correlation ID and all spans associated
    with processing a single user query from start to finish.
    
    Attributes:
        trace_id (str): Unique correlation ID for this request.
        start_time (float): Unix timestamp when the trace started.
        spans (List[Span]): List of all spans within this trace.
        metadata (Dict[str, Any]): Request-level metadata.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_span(self, span: Span) -> None:
        """
        Add a span to this trace.
        
        Args:
            span (Span): The span to add.
        """
        self.spans.append(span)
    
    def get_total_duration_ms(self) -> float:
        """
        Calculate the total duration of the trace in milliseconds.
        
        Returns:
            float: Total duration in milliseconds.
        """
        if not self.spans:
            return 0.0
        first_start = min(s.start_time for s in self.spans)
        last_end = max(s.end_time for s in self.spans if s.end_time)
        if last_end:
            return (last_end - first_start) * 1000
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the trace context to a dictionary for logging and serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the trace.
        """
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "start_time_iso": datetime.fromtimestamp(self.start_time).isoformat(),
            "total_duration_ms": round(self.get_total_duration_ms(), 2),
            "span_count": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata
        }


def start_trace(metadata: Optional[Dict[str, Any]] = None) -> TraceContext:
    """
    Start a new trace context for a request.
    
    This function creates a new trace context with a unique correlation ID
    and sets it as the current trace in the context variable.
    
    Args:
        metadata (Optional[Dict[str, Any]]): Initial metadata for the trace.
        
    Returns:
        TraceContext: The newly created trace context.
    """
    trace = TraceContext(metadata=metadata or {})
    _current_trace.set(trace)
    return trace


def get_current_trace() -> Optional[TraceContext]:
    """
    Get the current trace context from the context variable.
    
    Returns:
        Optional[TraceContext]: The current trace context, or None if no trace is active.
    """
    return _current_trace.get()


def end_trace() -> Optional[Dict[str, Any]]:
    """
    End the current trace and return its serialized form.
    
    This function clears the current trace from the context variable
    and returns the trace data for logging or storage.
    
    Returns:
        Optional[Dict[str, Any]]: The serialized trace data, or None if no trace was active.
    """
    trace = _current_trace.get()
    if trace:
        _current_trace.set(None)
        return trace.to_dict()
    return None


def start_span(operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> Span:
    """
    Start a new span within the current trace.
    
    If no trace is active, this creates an orphan span (useful for testing).
    
    Args:
        operation_name (str): Name of the operation being traced.
        metadata (Optional[Dict[str, Any]]): Initial metadata for the span.
        
    Returns:
        Span: The newly created span.
    """
    span = Span(
        operation_name=operation_name,
        metadata=metadata or {}
    )
    trace = get_current_trace()
    if trace:
        trace.add_span(span)
    return span


def complete_span(span: Span, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Mark a span as completed.
    
    Args:
        span (Span): The span to complete.
        metadata (Optional[Dict[str, Any]]): Additional metadata to merge.
    """
    span.complete(metadata)


def fail_span(span: Span, error_message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Mark a span as failed.
    
    Args:
        span (Span): The span to mark as failed.
        error_message (str): Description of the error.
        metadata (Optional[Dict[str, Any]]): Additional metadata to merge.
    """
    span.fail(error_message, metadata)


def get_trace_id() -> Optional[str]:
    """
    Get the current trace ID for logging purposes.
    
    Returns:
        Optional[str]: The current trace ID, or None if no trace is active.
    """
    trace = get_current_trace()
    return trace.trace_id if trace else None


class TracingContextManager:
    """
    Context manager for managing trace lifecycle.
    
    This class provides a convenient way to manage traces using the 'with' statement,
    ensuring proper cleanup even if exceptions occur.
    
    Example:
        with TracingContextManager({"user_query": "How many products?"}) as trace:
            # Operations within the trace
            pass
        # Trace is automatically ended and logged
    """
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the context manager.
        
        Args:
            metadata (Optional[Dict[str, Any]]): Initial metadata for the trace.
        """
        self.metadata = metadata
        self.trace: Optional[TraceContext] = None
    
    def __enter__(self) -> TraceContext:
        """
        Enter the context and start a new trace.
        
        Returns:
            TraceContext: The newly created trace context.
        """
        self.trace = start_trace(self.metadata)
        return self.trace
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context and end the trace.
        
        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        if exc_type and self.trace:
            self.trace.metadata["error"] = str(exc_val)
        end_trace()


class SpanContextManager:
    """
    Context manager for managing span lifecycle.
    
    This class provides a convenient way to manage spans using the 'with' statement,
    ensuring proper timing and status tracking even if exceptions occur.
    
    Example:
        with SpanContextManager("llm_call", {"model": "llama-3"}) as span:
            result = llm.invoke()
            span.metadata["tokens"] = result.token_count
        # Span is automatically completed or failed
    """
    
    def __init__(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the context manager.
        
        Args:
            operation_name (str): Name of the operation being traced.
            metadata (Optional[Dict[str, Any]]): Initial metadata for the span.
        """
        self.operation_name = operation_name
        self.metadata = metadata
        self.span: Optional[Span] = None
    
    def __enter__(self) -> Span:
        """
        Enter the context and start a new span.
        
        Returns:
            Span: The newly created span.
        """
        self.span = start_span(self.operation_name, self.metadata)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context and complete or fail the span.
        
        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        if self.span:
            if exc_type:
                fail_span(self.span, str(exc_val))
            else:
                complete_span(self.span)