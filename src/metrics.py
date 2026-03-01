"""
metrics.py

This module provides metrics collection capabilities for the Agentic Workflow application.
It tracks LLM-specific metrics (tokens, latency, cost) and tool execution metrics
(query timing, success rates, row counts) to enable performance monitoring and analysis.

The metrics system integrates with the tracing module to provide comprehensive
observability for production monitoring and debugging.
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


# Cost per 1K tokens (in USD) for common Groq models
# These are approximate values and should be updated based on current pricing
MODEL_COSTS = {
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama-3.1-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama-3.1-8b-instant": {"input": 0.00002, "output": 0.00002},
    "llama-3.2-1b-preview": {"input": 0.00004, "output": 0.00004},
    "llama-3.2-3b-preview": {"input": 0.00006, "output": 0.00006},
    "llama3-8b-8192": {"input": 0.00005, "output": 0.00008},
    "llama3-70b-8192": {"input": 0.00059, "output": 0.00079},
    "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
    "gemma2-9b-it": {"input": 0.00002, "output": 0.00002},
    # Default fallback for unknown models
    "default": {"input": 0.0001, "output": 0.0001}
}


@dataclass
class LLMCallMetrics:
    """
    Metrics for a single LLM API call.
    
    Captures token usage, latency, and estimated cost for each LLM invocation
    to enable cost tracking and performance analysis.
    
    Attributes:
        model (str): The model identifier used for the call.
        input_tokens (int): Number of tokens in the prompt.
        output_tokens (int): Number of tokens in the response.
        total_tokens (int): Total tokens used (input + output).
        latency_ms (float): Time taken for the API call in milliseconds.
        estimated_cost_usd (float): Estimated cost in USD.
        timestamp (float): Unix timestamp of the call.
        success (bool): Whether the call succeeded.
        error_message (Optional[str]): Error message if the call failed.
    """
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error_message: Optional[str] = None
    
    def calculate_cost(self) -> float:
        """
        Calculate the estimated cost based on token usage and model pricing.
        
        Returns:
            float: Estimated cost in USD.
        """
        pricing = MODEL_COSTS.get(self.model, MODEL_COSTS["default"])
        input_cost = (self.input_tokens / 1000) * pricing["input"]
        output_cost = (self.output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the metrics to a dictionary for logging and serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the metrics.
        """
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "success": self.success,
            "error_message": self.error_message
        }


@dataclass
class ToolCallMetrics:
    """
    Metrics for a single tool execution.
    
    Captures execution timing, result size, and success status for each tool call
    to enable performance monitoring and error tracking.
    
    Attributes:
        tool_name (str): Name of the tool that was executed.
        latency_ms (float): Time taken for execution in milliseconds.
        success (bool): Whether the execution succeeded.
        error_message (Optional[str]): Error message if execution failed.
        result_size_bytes (int): Size of the result in bytes.
        rows_affected (int): Number of rows affected or returned.
        timestamp (float): Unix timestamp of the call.
    """
    tool_name: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    result_size_bytes: int = 0
    rows_affected: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the metrics to a dictionary for logging and serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the metrics.
        """
        return {
            "tool_name": self.tool_name,
            "latency_ms": round(self.latency_ms, 2),
            "success": self.success,
            "error_message": self.error_message,
            "result_size_bytes": self.result_size_bytes,
            "rows_affected": self.rows_affected,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class QueryMetrics:
    """
    Aggregated metrics for a complete user query.
    
    Combines all LLM calls, tool calls, and overall timing for a single
    user interaction to provide a complete picture of query processing.
    
    Attributes:
        query_id (str): Unique identifier for this query (correlates with trace ID).
        user_query (str): The original user query text.
        total_latency_ms (float): Total time to process the query.
        llm_calls (List[LLMCallMetrics]): All LLM calls made during processing.
        tool_calls (List[ToolCallMetrics]): All tool calls made during processing.
        total_input_tokens (int): Sum of all input tokens.
        total_output_tokens (int): Sum of all output tokens.
        total_tokens (int): Sum of all tokens.
        total_cost_usd (float): Sum of all estimated costs.
        iterations (int): Number of agent iterations used.
        final_success (bool): Whether the query was answered successfully.
    """
    query_id: str = ""
    user_query: str = ""
    total_latency_ms: float = 0.0
    llm_calls: List[LLMCallMetrics] = field(default_factory=list)
    tool_calls: List[ToolCallMetrics] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    iterations: int = 0
    final_success: bool = True
    
    def add_llm_call(self, llm_metrics: LLMCallMetrics) -> None:
        """
        Add an LLM call metrics to this query.
        
        Args:
            llm_metrics (LLMCallMetrics): The LLM call metrics to add.
        """
        self.llm_calls.append(llm_metrics)
        self.total_input_tokens += llm_metrics.input_tokens
        self.total_output_tokens += llm_metrics.output_tokens
        self.total_tokens += llm_metrics.total_tokens
        self.total_cost_usd += llm_metrics.estimated_cost_usd
    
    def add_tool_call(self, tool_metrics: ToolCallMetrics) -> None:
        """
        Add a tool call metrics to this query.
        
        Args:
            tool_metrics (ToolCallMetrics): The tool call metrics to add.
        """
        self.tool_calls.append(tool_metrics)
    
    def finalize(self, success: bool = True) -> None:
        """
        Mark the query as complete and set final status.
        
        Args:
            success (bool): Whether the query was answered successfully.
        """
        self.final_success = success
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the metrics to a dictionary for logging and serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the metrics.
        """
        return {
            "query_id": self.query_id,
            "user_query": self.user_query[:100] + "..." if len(self.user_query) > 100 else self.user_query,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "llm_call_count": len(self.llm_calls),
            "tool_call_count": len(self.tool_calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "iterations": self.iterations,
            "final_success": self.final_success,
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "tool_calls": [c.to_dict() for c in self.tool_calls]
        }


class MetricsCollector:
    """
    Central collector for all metrics in the application.
    
    This class provides a singleton-like interface for collecting, aggregating,
    and reporting metrics across the application lifecycle. It maintains
    running totals and supports querying for statistics.
    
    Attributes:
        queries (List[QueryMetrics]): All query metrics collected.
        total_queries (int): Total number of queries processed.
        successful_queries (int): Number of successful queries.
        failed_queries (int): Number of failed queries.
        total_tokens_used (int): Total tokens consumed.
        total_cost_usd (float): Total estimated cost.
    """
    
    def __init__(self):
        """
        Initialize the metrics collector.
        """
        self.queries: List[QueryMetrics] = []
        self.total_queries: int = 0
        self.successful_queries: int = 0
        self.failed_queries: int = 0
        self.total_tokens_used: int = 0
        self.total_cost_usd: float = 0.0
        self._tool_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0, "success_count": 0, "total_latency_ms": 0.0
        })
    
    def start_query(self, query_id: str, user_query: str) -> QueryMetrics:
        """
        Start tracking a new query.
        
        Args:
            query_id (str): Unique identifier for this query.
            user_query (str): The user's query text.
            
        Returns:
            QueryMetrics: The newly created query metrics object.
        """
        metrics = QueryMetrics(
            query_id=query_id,
            user_query=user_query
        )
        return metrics
    
    def record_llm_call(
        self,
        query_metrics: QueryMetrics,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> LLMCallMetrics:
        """
        Record an LLM call within a query.
        
        Args:
            query_metrics (QueryMetrics): The parent query metrics.
            model (str): The model identifier.
            input_tokens (int): Number of input tokens.
            output_tokens (int): Number of output tokens.
            latency_ms (float): Latency in milliseconds.
            success (bool): Whether the call succeeded.
            error_message (Optional[str]): Error message if failed.
            
        Returns:
            LLMCallMetrics: The recorded LLM call metrics.
        """
        llm_metrics = LLMCallMetrics(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message
        )
        llm_metrics.estimated_cost_usd = llm_metrics.calculate_cost()
        query_metrics.add_llm_call(llm_metrics)
        return llm_metrics
    
    def record_tool_call(
        self,
        query_metrics: QueryMetrics,
        tool_name: str,
        latency_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
        result_size_bytes: int = 0,
        rows_affected: int = 0
    ) -> ToolCallMetrics:
        """
        Record a tool call within a query.
        
        Args:
            query_metrics (QueryMetrics): The parent query metrics.
            tool_name (str): Name of the tool.
            latency_ms (float): Latency in milliseconds.
            success (bool): Whether the call succeeded.
            error_message (Optional[str]): Error message if failed.
            result_size_bytes (int): Size of the result.
            rows_affected (int): Number of rows affected.
            
        Returns:
            ToolCallMetrics: The recorded tool call metrics.
        """
        tool_metrics = ToolCallMetrics(
            tool_name=tool_name,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            result_size_bytes=result_size_bytes,
            rows_affected=rows_affected
        )
        query_metrics.add_tool_call(tool_metrics)
        
        # Update tool statistics
        self._tool_stats[tool_name]["count"] += 1
        self._tool_stats[tool_name]["total_latency_ms"] += latency_ms
        if success:
            self._tool_stats[tool_name]["success_count"] += 1
        
        return tool_metrics
    
    def finalize_query(
        self,
        query_metrics: QueryMetrics,
        total_latency_ms: float,
        iterations: int,
        success: bool = True
    ) -> None:
        """
        Finalize a query and add it to the collection.
        
        Args:
            query_metrics (QueryMetrics): The query metrics to finalize.
            total_latency_ms (float): Total processing time.
            iterations (int): Number of iterations used.
            success (bool): Whether the query succeeded.
        """
        query_metrics.total_latency_ms = total_latency_ms
        query_metrics.iterations = iterations
        query_metrics.finalize(success)
        
        self.queries.append(query_metrics)
        self.total_queries += 1
        self.total_tokens_used += query_metrics.total_tokens
        self.total_cost_usd += query_metrics.total_cost_usd
        
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.
        
        Returns:
            Dict[str, Any]: Summary statistics.
        """
        success_rate = (
            self.successful_queries / self.total_queries * 100
            if self.total_queries > 0 else 0.0
        )
        
        avg_latency = 0.0
        avg_tokens = 0.0
        if self.queries:
            avg_latency = sum(q.total_latency_ms for q in self.queries) / len(self.queries)
            avg_tokens = sum(q.total_tokens for q in self.queries) / len(self.queries)
        
        tool_stats = {}
        for tool_name, stats in self._tool_stats.items():
            tool_stats[tool_name] = {
                "count": stats["count"],
                "success_count": stats["success_count"],
                "success_rate": round(stats["success_count"] / stats["count"] * 100, 2) if stats["count"] > 0 else 0.0,
                "avg_latency_ms": round(stats["total_latency_ms"] / stats["count"], 2) if stats["count"] > 0 else 0.0
            }
        
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate_percent": round(success_rate, 2),
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_tokens_per_query": round(avg_tokens, 2),
            "tool_statistics": tool_stats
        }
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent query metrics.
        
        Args:
            limit (int): Maximum number of queries to return.
            
        Returns:
            List[Dict[str, Any]]: List of recent query metrics.
        """
        recent = self.queries[-limit:] if len(self.queries) > limit else self.queries
        return [q.to_dict() for q in recent]
    
    def reset(self) -> None:
        """
        Reset all collected metrics.
        
        This is useful for testing or starting a fresh collection period.
        """
        self.queries.clear()
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        self._tool_stats.clear()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.
    
    Creates a new instance if one does not exist.
    
    Returns:
        MetricsCollector: The global metrics collector.
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics_collector() -> None:
    """
    Reset the global metrics collector.
    
    This is primarily useful for testing.
    """
    global _metrics_collector
    if _metrics_collector:
        _metrics_collector.reset()
    else:
        _metrics_collector = MetricsCollector()


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    
    Uses a simple heuristic of approximately 4 characters per token,
    which is a reasonable approximation for English text.
    
    Args:
        text (str): The text to estimate tokens for.
        
    Returns:
        int: Estimated number of tokens.
    """
    if not text:
        return 0
    # Simple heuristic: ~4 characters per token for English text
    return len(text) // 4 + 1


def calculate_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the estimated cost for a model call.
    
    Args:
        model (str): The model identifier.
        input_tokens (int): Number of input tokens.
        output_tokens (int): Number of output tokens.
        
    Returns:
        float: Estimated cost in USD.
    """
    pricing = MODEL_COSTS.get(model, MODEL_COSTS["default"])
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return input_cost + output_cost