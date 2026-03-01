"""
observability.py

This module provides a unified observability layer for the Agentic Workflow application.
It integrates tracing, metrics collection, and structured logging to provide comprehensive
monitoring capabilities for production environments.

The observability system coordinates between the tracing module (correlation IDs, spans),
the metrics module (LLM and tool metrics), and the logger module (structured output)
to provide a single cohesive interface for instrumentation.
"""

import time
import json
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps

from src.tracing import (
    start_trace, end_trace, start_span, complete_span, fail_span,
    get_trace_id, get_current_trace, TracingContextManager, SpanContextManager
)
from src.metrics import (
    get_metrics_collector, QueryMetrics, estimate_tokens
)


class ObservabilityContext:
    """
    Unified context for managing observability across a request lifecycle.
    
    This class coordinates tracing, metrics, and logging for a single user query,
    providing a convenient interface for instrumenting the agent execution flow.
    
    Attributes:
        trace_id (str): The correlation ID for this request.
        query_metrics (QueryMetrics): The metrics object for this query.
        logger (logging.Logger): The logger instance for this context.
        start_time (float): The start timestamp of this context.
        debug_mode (bool): Whether debug mode is enabled.
    """
    
    def __init__(
        self,
        user_query: str,
        logger: Optional[logging.Logger] = None,
        debug_mode: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the observability context.
        
        Args:
            user_query (str): The user's query text.
            logger (Optional[logging.Logger]): Logger instance to use.
            debug_mode (bool): Enable verbose debug output.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the trace.
        """
        self.logger = logger or logging.getLogger("AgenticWorkflow")
        self.debug_mode = debug_mode
        self.start_time = time.time()
        
        # Start tracing
        trace_metadata = metadata or {}
        trace_metadata["user_query"] = user_query[:200]  # Truncate for storage
        self._trace = start_trace(trace_metadata)
        self.trace_id = self._trace.trace_id
        
        # Start metrics collection
        collector = get_metrics_collector()
        self.query_metrics = collector.start_query(self.trace_id, user_query)
        
        self._iteration_count = 0
        self._finalized = False
        
        self._log_structured("INFO", "Observability context initialized", {
            "trace_id": self.trace_id,
            "debug_mode": self.debug_mode
        })
    
    def _log_structured(self, level: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured message with trace context.
        
        Args:
            level (str): Log level (DEBUG, INFO, WARNING, ERROR).
            message (str): The log message.
            data (Optional[Dict[str, Any]]): Additional structured data.
        """
        log_data = {
            "trace_id": self.trace_id,
            "message": message
        }
        if data:
            log_data.update(data)
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"[trace_id={self.trace_id}] {message} | {json.dumps(data) if data else ''}")
        
        if self.debug_mode:
            # In debug mode, also print to console for immediate visibility
            print(f"[{level}] [trace_id={self.trace_id}] {message}")
            if data:
                print(f"  Data: {json.dumps(data, indent=2, default=str)}")
    
    def start_iteration(self) -> int:
        """
        Start a new agent iteration.
        
        Returns:
            int: The current iteration number (1-indexed).
        """
        self._iteration_count += 1
        self._log_structured("INFO", f"Starting iteration {self._iteration_count}")
        return self._iteration_count
    
    def record_llm_call(
        self,
        model: str,
        input_text: str,
        output_text: str,
        latency_ms: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record an LLM API call with automatic token estimation.
        
        Args:
            model (str): The model identifier.
            input_text (str): The input prompt text.
            output_text (str): The output response text.
            latency_ms (float): Latency in milliseconds.
            success (bool): Whether the call succeeded.
            error_message (Optional[str]): Error message if failed.
            
        Returns:
            Dict[str, Any]: The recorded metrics as a dictionary.
        """
        input_tokens = estimate_tokens(input_text)
        output_tokens = estimate_tokens(output_text)
        
        collector = get_metrics_collector()
        llm_metrics = collector.record_llm_call(
            query_metrics=self.query_metrics,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message
        )
        
        self._log_structured("INFO", "LLM call recorded", {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(llm_metrics.estimated_cost_usd, 6)
        })
        
        return llm_metrics.to_dict()
    
    def record_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: str,
        latency_ms: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a tool execution with automatic metrics collection.
        
        Args:
            tool_name (str): Name of the tool.
            tool_args (Dict[str, Any]): Arguments passed to the tool.
            result (str): The result string.
            latency_ms (float): Latency in milliseconds.
            success (bool): Whether the execution succeeded.
            error_message (Optional[str]): Error message if failed.
            
        Returns:
            Dict[str, Any]: The recorded metrics as a dictionary.
        """
        result_size = len(result.encode('utf-8')) if result else 0
        
        collector = get_metrics_collector()
        tool_metrics = collector.record_tool_call(
            query_metrics=self.query_metrics,
            tool_name=tool_name,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            result_size_bytes=result_size
        )
        
        self._log_structured("INFO", "Tool call recorded", {
            "tool_name": tool_name,
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "result_size_bytes": result_size
        })
        
        return tool_metrics.to_dict()
    
    def finalize(self, success: bool = True, final_response: Optional[str] = None) -> Dict[str, Any]:
        """
        Finalize the observability context and return summary.
        
        Args:
            success (bool): Whether the query was answered successfully.
            final_response (Optional[str]): The final response text.
            
        Returns:
            Dict[str, Any]: Summary of the observability data.
        """
        if self._finalized:
            return self._get_summary()
        
        self._finalized = True
        total_latency_ms = (time.time() - self.start_time) * 1000
        
        # Finalize metrics
        collector = get_metrics_collector()
        collector.finalize_query(
            query_metrics=self.query_metrics,
            total_latency_ms=total_latency_ms,
            iterations=self._iteration_count,
            success=success
        )
        
        # End tracing
        trace_data = end_trace()
        
        summary = self._get_summary()
        summary["final_response_preview"] = final_response[:200] if final_response else None
        
        self._log_structured("INFO", "Observability context finalized", {
            "total_latency_ms": round(total_latency_ms, 2),
            "iterations": self._iteration_count,
            "success": success,
            "total_tokens": self.query_metrics.total_tokens,
            "total_cost_usd": round(self.query_metrics.total_cost_usd, 6)
        })
        
        return summary
    
    def _get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the observability data.
        
        Returns:
            Dict[str, Any]: Summary dictionary.
        """
        return {
            "trace_id": self.trace_id,
            "iterations": self._iteration_count,
            "total_latency_ms": round((time.time() - self.start_time) * 1000, 2),
            "llm_calls": len(self.query_metrics.llm_calls),
            "tool_calls": len(self.query_metrics.tool_calls),
            "total_tokens": self.query_metrics.total_tokens,
            "total_cost_usd": round(self.query_metrics.total_cost_usd, 6),
            "finalized": self._finalized
        }
    
    def export_debug_state(self) -> Dict[str, Any]:
        """
        Export the complete state for debugging purposes.
        
        Returns:
            Dict[str, Any]: Complete state dictionary.
        """
        trace = get_current_trace()
        return {
            "trace": trace.to_dict() if trace else None,
            "metrics": self.query_metrics.to_dict(),
            "summary": self._get_summary()
        }


class ObservabilityManager:
    """
    Manager class for creating and tracking observability contexts.
    
    This class provides factory methods for creating observability contexts
    and maintains references for debugging and reporting.
    
    Attributes:
        contexts (Dict[str, ObservabilityContext]): Active contexts by trace ID.
        logger (logging.Logger): The logger instance.
        debug_mode (bool): Global debug mode setting.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, debug_mode: bool = False):
        """
        Initialize the observability manager.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance to use.
            debug_mode (bool): Enable debug mode for all contexts.
        """
        self.contexts: Dict[str, ObservabilityContext] = {}
        self.logger = logger or logging.getLogger("AgenticWorkflow")
        self.debug_mode = debug_mode
    
    def create_context(
        self,
        user_query: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ObservabilityContext:
        """
        Create a new observability context for a query.
        
        Args:
            user_query (str): The user's query text.
            metadata (Optional[Dict[str, Any]]): Additional metadata.
            
        Returns:
            ObservabilityContext: The newly created context.
        """
        context = ObservabilityContext(
            user_query=user_query,
            logger=self.logger,
            debug_mode=self.debug_mode,
            metadata=metadata
        )
        self.contexts[context.trace_id] = context
        return context
    
    def get_context(self, trace_id: str) -> Optional[ObservabilityContext]:
        """
        Get an existing context by trace ID.
        
        Args:
            trace_id (str): The trace ID to look up.
            
        Returns:
            Optional[ObservabilityContext]: The context if found, None otherwise.
        """
        return self.contexts.get(trace_id)
    
    def remove_context(self, trace_id: str) -> None:
        """
        Remove a context from tracking.
        
        Args:
            trace_id (str): The trace ID of the context to remove.
        """
        self.contexts.pop(trace_id, None)
    
    def get_all_active_contexts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active contexts as dictionaries.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of trace_id to context summary.
        """
        return {
            trace_id: ctx._get_summary()
            for trace_id, ctx in self.contexts.items()
        }
    
    def get_global_metrics_summary(self) -> Dict[str, Any]:
        """
        Get the global metrics summary from the collector.
        
        Returns:
            Dict[str, Any]: Global metrics summary.
        """
        collector = get_metrics_collector()
        return collector.get_summary()


# Global observability manager instance
_observability_manager: Optional[ObservabilityManager] = None


def get_observability_manager(
    logger: Optional[logging.Logger] = None,
    debug_mode: bool = False
) -> ObservabilityManager:
    """
    Get the global observability manager instance.
    
    Args:
        logger (Optional[logging.Logger]): Logger instance to use.
        debug_mode (bool): Enable debug mode.
        
    Returns:
        ObservabilityManager: The global observability manager.
    """
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager(logger=logger, debug_mode=debug_mode)
    return _observability_manager


def create_observability_context(
    user_query: str,
    logger: Optional[logging.Logger] = None,
    debug_mode: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> ObservabilityContext:
    """
    Convenience function to create an observability context.
    
    Args:
        user_query (str): The user's query text.
        logger (Optional[logging.Logger]): Logger instance to use.
        debug_mode (bool): Enable debug mode.
        metadata (Optional[Dict[str, Any]]): Additional metadata.
        
    Returns:
        ObservabilityContext: The newly created context.
    """
    manager = get_observability_manager(logger=logger, debug_mode=debug_mode)
    return manager.create_context(user_query=user_query, metadata=metadata)


def instrument_tool(func: Callable) -> Callable:
    """
    Decorator to automatically instrument a tool function with observability.
    
    This decorator wraps a tool function to automatically record timing,
    success/failure status, and result size.
    
    Args:
        func (Callable): The tool function to instrument.
        
    Returns:
        Callable: The wrapped function with observability.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        tool_name = func.__name__
        error_message = None
        success = True
        result = None
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            
            # Try to get current observability context
            trace = get_current_trace()
            if trace:
                # Create a span for this tool call
                span = start_span(f"tool_{tool_name}", {
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                })
                
                if success:
                    complete_span(span, {
                        "latency_ms": latency_ms,
                        "result_size": len(str(result)) if result else 0
                    })
                else:
                    fail_span(span, error_message or "Unknown error")
        
        return result
    
    return wrapper


def instrument_llm_call(func: Callable) -> Callable:
    """
    Decorator to automatically instrument an LLM call with observability.
    
    This decorator wraps an LLM invocation to automatically record timing,
    token estimation, and cost tracking.
    
    Args:
        func (Callable): The LLM call function to instrument.
        
    Returns:
        Callable: The wrapped function with observability.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        error_message = None
        success = True
        result = None
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            
            # Try to get current trace
            trace = get_current_trace()
            if trace:
                span = start_span("llm_call", {
                    "function": func.__name__
                })
                
                if success:
                    complete_span(span, {
                        "latency_ms": latency_ms
                    })
                else:
                    fail_span(span, error_message or "Unknown error")
        
        return result
    
    return wrapper