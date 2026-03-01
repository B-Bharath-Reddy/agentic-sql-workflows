"""
Tests for the metrics module.

This module tests the metrics collection functionality including
LLM call metrics, tool call metrics, and query metrics aggregation.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from src.metrics import (
    LLMCallMetrics, ToolCallMetrics, QueryMetrics,
    MetricsCollector, get_metrics_collector, reset_metrics_collector,
    estimate_tokens, calculate_model_cost, MODEL_COSTS
)


class TestLLMCallMetrics:
    """Tests for the LLMCallMetrics class."""
    
    def test_llm_call_metrics_creation(self):
        """Test creating LLM call metrics."""
        metrics = LLMCallMetrics(
            model="llama-3.3-70b-versatile",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500.0,
            estimated_cost_usd=0.0001,
            success=True
        )
        
        assert metrics.model == "llama-3.3-70b-versatile"
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.latency_ms == 500.0
        assert metrics.success is True
        assert metrics.error_message is None
    
    def test_llm_call_metrics_with_error(self):
        """Test LLM call metrics with error."""
        metrics = LLMCallMetrics(
            model="llama-3.3-70b-versatile",
            success=False,
            error_message="Rate limit exceeded"
        )
        
        assert metrics.success is False
        assert metrics.error_message == "Rate limit exceeded"
    
    def test_calculate_cost(self):
        """Test cost calculation."""
        metrics = LLMCallMetrics(
            model="llama-3.3-70b-versatile",
            input_tokens=1000,
            output_tokens=500
        )
        
        cost = metrics.calculate_cost()
        
        # Based on MODEL_COSTS: input: 0.00059, output: 0.00079 per 1K tokens
        expected_input_cost = 1.0 * 0.00059  # 1K input tokens
        expected_output_cost = 0.5 * 0.00079  # 0.5K output tokens
        expected_cost = expected_input_cost + expected_output_cost
        
        assert abs(cost - expected_cost) < 0.000001
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model uses default."""
        metrics = LLMCallMetrics(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=1000
        )
        
        cost = metrics.calculate_cost()
        
        # Should use default pricing
        expected_cost = 1.0 * 0.0001 + 1.0 * 0.0001
        assert abs(cost - expected_cost) < 0.000001
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = LLMCallMetrics(
            model="llama-3.3-70b-versatile",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500.0,
            estimated_cost_usd=0.0001,
            success=True
        )
        
        result = metrics.to_dict()
        
        assert result["model"] == "llama-3.3-70b-versatile"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["latency_ms"] == 500.0
        assert result["success"] is True
        assert "timestamp_iso" in result


class TestToolCallMetrics:
    """Tests for the ToolCallMetrics class."""
    
    def test_tool_call_metrics_creation(self):
        """Test creating tool call metrics."""
        metrics = ToolCallMetrics(
            tool_name="execute_sql_query",
            latency_ms=150.0,
            success=True,
            result_size_bytes=1024,
            rows_affected=10
        )
        
        assert metrics.tool_name == "execute_sql_query"
        assert metrics.latency_ms == 150.0
        assert metrics.success is True
        assert metrics.result_size_bytes == 1024
        assert metrics.rows_affected == 10
    
    def test_tool_call_metrics_with_error(self):
        """Test tool call metrics with error."""
        metrics = ToolCallMetrics(
            tool_name="execute_sql_query",
            success=False,
            error_message="Syntax error in SQL"
        )
        
        assert metrics.success is False
        assert metrics.error_message == "Syntax error in SQL"
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = ToolCallMetrics(
            tool_name="execute_sql_query",
            latency_ms=150.0,
            success=True,
            rows_affected=5
        )
        
        result = metrics.to_dict()
        
        assert result["tool_name"] == "execute_sql_query"
        assert result["latency_ms"] == 150.0
        assert result["success"] is True
        assert result["rows_affected"] == 5
        assert "timestamp_iso" in result


class TestQueryMetrics:
    """Tests for the QueryMetrics class."""
    
    def test_query_metrics_creation(self):
        """Test creating query metrics."""
        metrics = QueryMetrics(
            query_id="query-123",
            user_query="Show me all customers"
        )
        
        assert metrics.query_id == "query-123"
        assert metrics.user_query == "Show me all customers"
        assert metrics.llm_calls == []
        assert metrics.tool_calls == []
        assert metrics.total_tokens == 0
    
    def test_add_llm_call(self):
        """Test adding an LLM call."""
        metrics = QueryMetrics(query_id="query-123")
        llm_metrics = LLMCallMetrics(
            model="llama-3.3-70b-versatile",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.0001
        )
        
        metrics.add_llm_call(llm_metrics)
        
        assert len(metrics.llm_calls) == 1
        assert metrics.total_input_tokens == 100
        assert metrics.total_output_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.total_cost_usd == 0.0001
    
    def test_add_tool_call(self):
        """Test adding a tool call."""
        metrics = QueryMetrics(query_id="query-123")
        tool_metrics = ToolCallMetrics(
            tool_name="execute_sql_query",
            latency_ms=100.0
        )
        
        metrics.add_tool_call(tool_metrics)
        
        assert len(metrics.tool_calls) == 1
    
    def test_finalize(self):
        """Test finalizing query metrics."""
        metrics = QueryMetrics(query_id="query-123")
        
        metrics.finalize(success=True)
        
        assert metrics.final_success is True
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = QueryMetrics(
            query_id="query-123",
            user_query="Show me all customers",
            total_latency_ms=1000.0,
            iterations=2
        )
        metrics.add_llm_call(LLMCallMetrics(
            model="llama-3.3-70b-versatile",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150
        ))
        metrics.finalize(True)
        
        result = metrics.to_dict()
        
        assert result["query_id"] == "query-123"
        assert result["user_query"] == "Show me all customers"
        assert result["total_latency_ms"] == 1000.0
        assert result["llm_call_count"] == 1
        assert result["iterations"] == 2
        assert result["final_success"] is True
    
    def test_to_dict_truncates_long_query(self):
        """Test that long queries are truncated in to_dict."""
        long_query = "x" * 200
        metrics = QueryMetrics(
            query_id="query-123",
            user_query=long_query
        )
        
        result = metrics.to_dict()
        
        assert len(result["user_query"]) == 103  # 100 chars + "..."


class TestMetricsCollector:
    """Tests for the MetricsCollector class."""
    
    def setup_method(self):
        """Reset metrics collector before each test."""
        reset_metrics_collector()
    
    def test_start_query(self):
        """Test starting a query."""
        collector = MetricsCollector()
        
        metrics = collector.start_query("query-123", "Test query")
        
        assert metrics.query_id == "query-123"
        assert metrics.user_query == "Test query"
    
    def test_record_llm_call(self):
        """Test recording an LLM call."""
        collector = MetricsCollector()
        query_metrics = collector.start_query("query-123", "Test query")
        
        llm_metrics = collector.record_llm_call(
            query_metrics=query_metrics,
            model="llama-3.3-70b-versatile",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500.0
        )
        
        assert llm_metrics.model == "llama-3.3-70b-versatile"
        assert llm_metrics.input_tokens == 100
        assert llm_metrics.output_tokens == 50
        assert len(query_metrics.llm_calls) == 1
    
    def test_record_tool_call(self):
        """Test recording a tool call."""
        collector = MetricsCollector()
        query_metrics = collector.start_query("query-123", "Test query")
        
        tool_metrics = collector.record_tool_call(
            query_metrics=query_metrics,
            tool_name="execute_sql_query",
            latency_ms=150.0,
            rows_affected=10
        )
        
        assert tool_metrics.tool_name == "execute_sql_query"
        assert tool_metrics.latency_ms == 150.0
        assert len(query_metrics.tool_calls) == 1
    
    def test_finalize_query(self):
        """Test finalizing a query."""
        collector = MetricsCollector()
        query_metrics = collector.start_query("query-123", "Test query")
        
        collector.record_llm_call(
            query_metrics=query_metrics,
            model="llama-3.3-70b-versatile",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500.0
        )
        
        collector.finalize_query(
            query_metrics=query_metrics,
            total_latency_ms=1000.0,
            iterations=2,
            success=True
        )
        
        assert collector.total_queries == 1
        assert collector.successful_queries == 1
        assert collector.failed_queries == 0
        assert len(collector.queries) == 1
    
    def test_finalize_query_failure(self):
        """Test finalizing a failed query."""
        collector = MetricsCollector()
        query_metrics = collector.start_query("query-123", "Test query")
        
        collector.finalize_query(
            query_metrics=query_metrics,
            total_latency_ms=1000.0,
            iterations=2,
            success=False
        )
        
        assert collector.total_queries == 1
        assert collector.successful_queries == 0
        assert collector.failed_queries == 1
    
    def test_get_summary(self):
        """Test getting summary statistics."""
        collector = MetricsCollector()
        
        # Add a successful query
        query1 = collector.start_query("query-1", "Test query 1")
        collector.record_llm_call(query1, "llama-3.3-70b-versatile", 100, 50, 500.0)
        collector.finalize_query(query1, 1000.0, 1, success=True)
        
        # Add a failed query
        query2 = collector.start_query("query-2", "Test query 2")
        collector.finalize_query(query2, 500.0, 1, success=False)
        
        summary = collector.get_summary()
        
        assert summary["total_queries"] == 2
        assert summary["successful_queries"] == 1
        assert summary["failed_queries"] == 1
        assert summary["success_rate_percent"] == 50.0
        assert summary["total_tokens_used"] == 150
    
    def test_get_recent_queries(self):
        """Test getting recent queries."""
        collector = MetricsCollector()
        
        for i in range(15):
            query = collector.start_query(f"query-{i}", f"Test query {i}")
            collector.finalize_query(query, 100.0, 1, success=True)
        
        recent = collector.get_recent_queries(limit=10)
        
        assert len(recent) == 10
        # Should be the most recent ones
        assert recent[-1]["query_id"] == "query-14"
    
    def test_reset(self):
        """Test resetting the collector."""
        collector = MetricsCollector()
        
        query = collector.start_query("query-1", "Test query")
        collector.finalize_query(query, 100.0, 1, success=True)
        
        assert collector.total_queries == 1
        
        collector.reset()
        
        assert collector.total_queries == 0
        assert len(collector.queries) == 0
    
    def test_tool_statistics(self):
        """Test tool statistics tracking."""
        collector = MetricsCollector()
        
        query = collector.start_query("query-1", "Test query")
        collector.record_tool_call(query, "execute_sql_query", 100.0, success=True)
        collector.record_tool_call(query, "execute_sql_query", 200.0, success=True)
        collector.record_tool_call(query, "execute_sql_query", 300.0, success=False)
        collector.finalize_query(query, 500.0, 1, success=True)
        
        summary = collector.get_summary()
        tool_stats = summary["tool_statistics"]["execute_sql_query"]
        
        assert tool_stats["count"] == 3
        assert tool_stats["success_count"] == 2
        assert tool_stats["success_rate"] == pytest.approx(66.67, rel=0.01)
        assert tool_stats["avg_latency_ms"] == 200.0


class TestGetMetricsCollector:
    """Tests for the get_metrics_collector function."""
    
    def setup_method(self):
        """Reset before each test."""
        global _metrics_collector
        import src.metrics as metrics_module
        metrics_module._metrics_collector = None
    
    def test_returns_singleton(self):
        """Test that get_metrics_collector returns a singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2
    
    def test_reset_metrics_collector(self):
        """Test resetting the global collector."""
        collector = get_metrics_collector()
        query = collector.start_query("query-1", "Test")
        collector.finalize_query(query, 100.0, 1, True)
        
        assert collector.total_queries == 1
        
        reset_metrics_collector()
        
        assert collector.total_queries == 0


class TestEstimateTokens:
    """Tests for the estimate_tokens function."""
    
    def test_empty_string(self):
        """Test estimating tokens for empty string."""
        assert estimate_tokens("") == 0
    
    def test_short_string(self):
        """Test estimating tokens for short string."""
        # ~4 chars per token
        result = estimate_tokens("hello")
        assert result >= 1
    
    def test_longer_string(self):
        """Test estimating tokens for longer string."""
        text = "a" * 100
        result = estimate_tokens(text)
        # Should be approximately 25 tokens (100/4)
        assert 20 <= result <= 30


class TestCalculateModelCost:
    """Tests for the calculate_model_cost function."""
    
    def test_calculate_cost_known_model(self):
        """Test cost calculation for known model."""
        cost = calculate_model_cost(
            model="llama-3.3-70b-versatile",
            input_tokens=1000,
            output_tokens=500
        )
        
        # Based on MODEL_COSTS
        expected = 1.0 * 0.00059 + 0.5 * 0.00079
        assert abs(cost - expected) < 0.000001
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model uses default."""
        cost = calculate_model_cost(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=1000
        )
        
        # Should use default pricing
        expected = 1.0 * 0.0001 + 1.0 * 0.0001
        assert abs(cost - expected) < 0.000001


class TestModelCosts:
    """Tests for MODEL_COSTS configuration."""
    
    def test_model_costs_have_required_keys(self):
        """Test that all model costs have input and output keys."""
        for model, pricing in MODEL_COSTS.items():
            assert "input" in pricing, f"Missing 'input' for {model}"
            assert "output" in pricing, f"Missing 'output' for {model}"
            assert pricing["input"] >= 0, f"Invalid input cost for {model}"
            assert pricing["output"] >= 0, f"Invalid output cost for {model}"
    
    def test_default_model_exists(self):
        """Test that default model pricing exists."""
        assert "default" in MODEL_COSTS