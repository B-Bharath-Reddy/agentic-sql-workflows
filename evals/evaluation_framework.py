"""
evaluation_framework.py

A comprehensive evaluation framework for the Agentic SQL Workflow application.
This module provides structured evaluation capabilities including:

1. Component-Level Evaluation: Tests intermediate outputs (planner, tool calling)
2. System-Level Evaluation: End-to-end black box testing
3. Performance Metrics: Latency, token usage, cost tracking
4. Regression Testing: Ensure changes don't break existing functionality
"""

import json
import os
import time
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from pathlib import Path

# Import observability for metrics tracking
from src.observability import create_observability_context
from src.metrics import get_metrics_collector, estimate_tokens
from src.tracing import start_trace, end_trace, start_span, complete_span
from src.error_handling import ErrorHandler, ErrorCategory


class EvaluationStatus(Enum):
    """Status of an evaluation test."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class EvaluationType(Enum):
    """Types of evaluations."""
    COMPONENT = "component"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    REGRESSION = "regression"


@dataclass
class EvaluationResult:
    """Result of a single evaluation test."""
    test_name: str
    evaluation_type: EvaluationType
    status: EvaluationStatus
    message: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "evaluation_type": self.evaluation_type.value,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "details": self.details,
            "error": str(self.error) if self.error else None,
            "timestamp": self.timestamp
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report with all results."""
    name: str
    results: List[EvaluationResult] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == EvaluationStatus.PASSED)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == EvaluationStatus.FAILED)
    
    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == EvaluationStatus.ERROR)
    
    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == EvaluationStatus.SKIPPED)
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100
    
    def add_result(self, result: EvaluationResult):
        """Add a result to the report."""
        self.results.append(result)
    
    def finalize(self):
        """Mark the report as complete."""
        self.end_time = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "skipped": self.skipped,
                "pass_rate": round(self.pass_rate, 2)
            },
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata
        }
    
    def save(self, path: str):
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_summary(self):
        """Print a summary of the report."""
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {self.name}")
        print(f"{'='*60}")
        print(f"Total Tests: {self.total}")
        print(f"Passed: {self.passed} | Failed: {self.failed} | Errors: {self.errors} | Skipped: {self.skipped}")
        print(f"Pass Rate: {self.pass_rate:.1f}%")
        print(f"{'='*60}\n")
        
        # Print failed tests
        failed_tests = [r for r in self.results if r.status in (EvaluationStatus.FAILED, EvaluationStatus.ERROR)]
        if failed_tests:
            print("Failed Tests:")
            for r in failed_tests:
                print(f"  - {r.test_name}: {r.message}")
            print()


class Evaluator:
    """Base evaluator class with common functionality."""
    
    def __init__(self, name: str = "Evaluation"):
        self.name = name
        self.error_handler = ErrorHandler()
    
    def create_report(self, metadata: Dict[str, Any] = None) -> EvaluationReport:
        """Create a new evaluation report."""
        return EvaluationReport(
            name=self.name,
            metadata=metadata or {}
        )
    
    def evaluate(
        self,
        test_name: str,
        evaluation_type: EvaluationType,
        test_func: Callable,
        *args,
        **kwargs
    ) -> EvaluationResult:
        """Run a single evaluation test."""
        start_time = time.time()
        
        try:
            result = test_func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, bool):
                status = EvaluationStatus.PASSED if result else EvaluationStatus.FAILED
                return EvaluationResult(
                    test_name=test_name,
                    evaluation_type=evaluation_type,
                    status=status,
                    message="Test passed" if result else "Test failed",
                    duration_ms=duration_ms
                )
            elif isinstance(result, EvaluationResult):
                result.duration_ms = duration_ms
                return result
            elif isinstance(result, dict):
                return EvaluationResult(
                    test_name=test_name,
                    evaluation_type=evaluation_type,
                    status=EvaluationStatus.PASSED if result.get("passed", False) else EvaluationStatus.FAILED,
                    message=result.get("message", ""),
                    duration_ms=duration_ms,
                    details=result.get("details", {})
                )
            else:
                return EvaluationResult(
                    test_name=test_name,
                    evaluation_type=evaluation_type,
                    status=EvaluationStatus.PASSED,
                    message="Test completed",
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.error_handler.handle_error(e, {"test_name": test_name})
            return EvaluationResult(
                test_name=test_name,
                evaluation_type=evaluation_type,
                status=EvaluationStatus.ERROR,
                message=str(e),
                duration_ms=duration_ms,
                error=e
            )


class ComponentEvaluator(Evaluator):
    """
    Component-level evaluator for testing intermediate outputs.
    
    Tests the "Planner" and "Tool Calling" components by intercepting
    the LLM's first thought to ensure it selects correct tools and
    understands the schema.
    """
    
    def __init__(self):
        super().__init__("Component Evaluation")
    
    def evaluate_tool_selection(
        self,
        query: str,
        expected_tools: List[str],
        agent: Any = None
    ) -> EvaluationResult:
        """
        Evaluate if the agent selects the correct tools for a query.
        
        Args:
            query: The user query
            expected_tools: List of expected tool names
            agent: The agent instance (optional, will create if not provided)
        
        Returns:
            EvaluationResult with tool selection details
        """
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            if agent is None:
                from src.agent_core import CustomerServiceAgent
                agent = CustomerServiceAgent()
            
            # Get the initial LLM response (before tool execution)
            from langchain_core.messages import SystemMessage, HumanMessage
            
            system_prompt = agent._get_system_prompt()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            # Invoke LLM once to get tool calls
            response = agent.llm_with_tools.invoke(messages)
            
            # Extract called tools
            called_tools = []
            if hasattr(response, 'tool_calls') and response.tool_calls:
                called_tools = [tc["name"] for tc in response.tool_calls]
            
            # Check if expected tools were called
            missing_tools = set(expected_tools) - set(called_tools)
            unexpected_tools = set(called_tools) - set(expected_tools)
            
            passed = len(missing_tools) == 0
            duration_ms = (time.time() - start_time) * 1000
            
            return EvaluationResult(
                test_name=f"tool_selection_{query[:30]}",
                evaluation_type=EvaluationType.COMPONENT,
                status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
                message=f"Expected tools: {expected_tools}, Called: {called_tools}",
                duration_ms=duration_ms,
                details={
                    "query": query,
                    "expected_tools": expected_tools,
                    "called_tools": called_tools,
                    "missing_tools": list(missing_tools),
                    "unexpected_tools": list(unexpected_tools)
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return EvaluationResult(
                test_name=f"tool_selection_{query[:30]}",
                evaluation_type=EvaluationType.COMPONENT,
                status=EvaluationStatus.ERROR,
                message=str(e),
                duration_ms=duration_ms,
                error=e
            )
    
    def evaluate_sql_generation(
        self,
        query: str,
        expected_table: str,
        agent: Any = None
    ) -> EvaluationResult:
        """
        Evaluate if the agent generates SQL targeting the correct table.
        
        Args:
            query: The user query
            expected_table: Expected table name in SQL
            agent: The agent instance
        
        Returns:
            EvaluationResult with SQL generation details
        """
        start_time = time.time()
        
        try:
            if agent is None:
                from src.agent_core import CustomerServiceAgent
                agent = CustomerServiceAgent()
            
            # Run the agent and capture SQL queries
            from langchain_core.messages import SystemMessage, HumanMessage
            
            system_prompt = agent._get_system_prompt()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            # Track SQL queries
            sql_queries = []
            
            # Simulate agent execution
            response = agent.llm_with_tools.invoke(messages)
            
            if hasattr(response, 'tool_calls'):
                for tc in response.tool_calls:
                    if tc["name"] == "execute_sql_query":
                        sql = tc["args"].get("query", "")
                        sql_queries.append(sql)
            
            # Check if expected table is in any SQL query
            found_table = any(expected_table.lower() in sql.lower() for sql in sql_queries)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return EvaluationResult(
                test_name=f"sql_generation_{expected_table}",
                evaluation_type=EvaluationType.COMPONENT,
                status=EvaluationStatus.PASSED if found_table else EvaluationStatus.FAILED,
                message=f"Expected table '{expected_table}' {'found' if found_table else 'not found'} in SQL",
                duration_ms=duration_ms,
                details={
                    "query": query,
                    "expected_table": expected_table,
                    "sql_queries": sql_queries
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return EvaluationResult(
                test_name=f"sql_generation_{expected_table}",
                evaluation_type=EvaluationType.COMPONENT,
                status=EvaluationStatus.ERROR,
                message=str(e),
                duration_ms=duration_ms,
                error=e
            )


class SystemEvaluator(Evaluator):
    """
    System-level evaluator for end-to-end black box testing.
    
    Tests the complete agent execution to ensure final responses
    contain factually correct answers.
    """
    
    def __init__(self):
        super().__init__("System Evaluation")
    
    def evaluate_end_to_end(
        self,
        query: str,
        expected_answer: str,
        agent: Any = None,
        case_sensitive: bool = False
    ) -> EvaluationResult:
        """
        Evaluate end-to-end agent response.
        
        Args:
            query: The user query
            expected_answer: Expected answer in the response
            agent: The agent instance
            case_sensitive: Whether to do case-sensitive matching
        
        Returns:
            EvaluationResult with end-to-end details
        """
        start_time = time.time()
        
        try:
            if agent is None:
                from src.agent_core import CustomerServiceAgent
                agent = CustomerServiceAgent()
            
            # Run the full agent loop
            response = agent.run(query)
            
            # Check if expected answer is in response
            response_check = response if case_sensitive else response.lower()
            answer_check = expected_answer if case_sensitive else expected_answer.lower()
            
            passed = answer_check in response_check
            duration_ms = (time.time() - start_time) * 1000
            
            return EvaluationResult(
                test_name=f"e2e_{query[:30]}",
                evaluation_type=EvaluationType.SYSTEM,
                status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
                message=f"Expected '{expected_answer}' {'found' if passed else 'not found'} in response",
                duration_ms=duration_ms,
                details={
                    "query": query,
                    "expected_answer": expected_answer,
                    "response": response[:500] if len(response) > 500 else response,
                    "response_length": len(response)
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return EvaluationResult(
                test_name=f"e2e_{query[:30]}",
                evaluation_type=EvaluationType.SYSTEM,
                status=EvaluationStatus.ERROR,
                message=str(e),
                duration_ms=duration_ms,
                error=e
            )
    
    def evaluate_response_quality(
        self,
        query: str,
        response: str,
        criteria: Dict[str, Any] = None
    ) -> EvaluationResult:
        """
        Evaluate the quality of a response based on criteria.
        
        Args:
            query: The user query
            response: The agent's response
            criteria: Quality criteria (min_length, max_length, required_terms, etc.)
        
        Returns:
            EvaluationResult with quality assessment
        """
        start_time = time.time()
        criteria = criteria or {}
        
        issues = []
        
        # Check minimum length
        min_length = criteria.get("min_length", 10)
        if len(response) < min_length:
            issues.append(f"Response too short ({len(response)} < {min_length})")
        
        # Check maximum length
        max_length = criteria.get("max_length", 10000)
        if len(response) > max_length:
            issues.append(f"Response too long ({len(response)} > {max_length})")
        
        # Check required terms
        required_terms = criteria.get("required_terms", [])
        for term in required_terms:
            if term.lower() not in response.lower():
                issues.append(f"Missing required term: '{term}'")
        
        # Check forbidden terms
        forbidden_terms = criteria.get("forbidden_terms", [])
        for term in forbidden_terms:
            if term.lower() in response.lower():
                issues.append(f"Contains forbidden term: '{term}'")
        
        # Check for error indicators
        error_indicators = criteria.get("error_indicators", ["error", "failed", "exception", "sorry"])
        for indicator in error_indicators:
            if indicator.lower() in response.lower():
                issues.append(f"Contains error indicator: '{indicator}'")
        
        passed = len(issues) == 0
        duration_ms = (time.time() - start_time) * 1000
        
        return EvaluationResult(
            test_name=f"quality_{query[:30]}",
            evaluation_type=EvaluationType.SYSTEM,
            status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
            message="Quality check passed" if passed else "; ".join(issues),
            duration_ms=duration_ms,
            details={
                "query": query,
                "issues": issues,
                "criteria": criteria
            }
        )


class PerformanceEvaluator(Evaluator):
    """
    Performance evaluator for measuring latency, throughput, and resource usage.
    """
    
    def __init__(self):
        super().__init__("Performance Evaluation")
    
    def evaluate_latency(
        self,
        query: str,
        max_latency_ms: float,
        agent: Any = None
    ) -> EvaluationResult:
        """
        Evaluate if the agent responds within acceptable latency.
        
        Args:
            query: The user query
            max_latency_ms: Maximum acceptable latency in milliseconds
            agent: The agent instance
        
        Returns:
            EvaluationResult with latency details
        """
        start_time = time.time()
        
        try:
            if agent is None:
                from src.agent_core import CustomerServiceAgent
                agent = CustomerServiceAgent()
            
            # Run agent and measure time
            response = agent.run(query)
            actual_latency_ms = (time.time() - start_time) * 1000
            
            passed = actual_latency_ms <= max_latency_ms
            
            return EvaluationResult(
                test_name=f"latency_{query[:30]}",
                evaluation_type=EvaluationType.PERFORMANCE,
                status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
                message=f"Latency: {actual_latency_ms:.0f}ms (max: {max_latency_ms}ms)",
                duration_ms=actual_latency_ms,
                details={
                    "query": query,
                    "actual_latency_ms": actual_latency_ms,
                    "max_latency_ms": max_latency_ms,
                    "response_length": len(response)
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return EvaluationResult(
                test_name=f"latency_{query[:30]}",
                evaluation_type=EvaluationType.PERFORMANCE,
                status=EvaluationStatus.ERROR,
                message=str(e),
                duration_ms=duration_ms,
                error=e
            )
    
    def benchmark_queries(
        self,
        queries: List[str],
        agent: Any = None,
        warmup_runs: int = 1
    ) -> EvaluationResult:
        """
        Benchmark multiple queries and return performance statistics.
        
        Args:
            queries: List of queries to benchmark
            agent: The agent instance
            warmup_runs: Number of warmup runs before measuring
        
        Returns:
            EvaluationResult with benchmark statistics
        """
        start_time = time.time()
        
        try:
            if agent is None:
                from src.agent_core import CustomerServiceAgent
                agent = CustomerServiceAgent()
            
            # Warmup runs
            if warmup_runs > 0 and queries:
                for _ in range(warmup_runs):
                    agent.run(queries[0])
            
            # Benchmark runs
            latencies = []
            for query in queries:
                query_start = time.time()
                agent.run(query)
                latencies.append((time.time() - query_start) * 1000)
            
            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return EvaluationResult(
                test_name="benchmark_queries",
                evaluation_type=EvaluationType.PERFORMANCE,
                status=EvaluationStatus.PASSED,
                message=f"Benchmarked {len(queries)} queries",
                duration_ms=duration_ms,
                details={
                    "query_count": len(queries),
                    "avg_latency_ms": round(avg_latency, 2),
                    "min_latency_ms": round(min_latency, 2),
                    "max_latency_ms": round(max_latency, 2),
                    "latencies": latencies
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return EvaluationResult(
                test_name="benchmark_queries",
                evaluation_type=EvaluationType.PERFORMANCE,
                status=EvaluationStatus.ERROR,
                message=str(e),
                duration_ms=duration_ms,
                error=e
            )


class RegressionEvaluator(Evaluator):
    """
    Regression evaluator for ensuring changes don't break existing functionality.
    """
    
    def __init__(self, baseline_path: str = None):
        super().__init__("Regression Evaluation")
        self.baseline_path = baseline_path or os.path.join(os.path.dirname(__file__), "baseline.json")
    
    def create_baseline(
        self,
        queries: List[Dict[str, str]],
        agent: Any = None
    ) -> Dict[str, Any]:
        """
        Create a baseline of expected responses.
        
        Args:
            queries: List of dicts with 'query' and 'name' keys
            agent: The agent instance
        
        Returns:
            Baseline dictionary
        """
        if agent is None:
            from src.agent_core import CustomerServiceAgent
            agent = CustomerServiceAgent()
        
        baseline = {
            "created_at": datetime.now().isoformat(),
            "queries": {}
        }
        
        for item in queries:
            query = item["query"]
            name = item["name"]
            
            start_time = time.time()
            response = agent.run(query)
            duration_ms = (time.time() - start_time) * 1000
            
            baseline["queries"][name] = {
                "query": query,
                "response_hash": hash(response),
                "response_length": len(response),
                "duration_ms": duration_ms
            }
        
        # Save baseline
        with open(self.baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        return baseline
    
    def load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline from file."""
        if os.path.exists(self.baseline_path):
            with open(self.baseline_path, 'r') as f:
                return json.load(f)
        return None
    
    def evaluate_regression(
        self,
        agent: Any = None,
        tolerance_ms: float = 5000
    ) -> EvaluationResult:
        """
        Evaluate against baseline to detect regressions.
        
        Args:
            agent: The agent instance
            tolerance_ms: Acceptable latency increase in ms
        
        Returns:
            EvaluationResult with regression details
        """
        start_time = time.time()
        
        baseline = self.load_baseline()
        if not baseline:
            return EvaluationResult(
                test_name="regression_check",
                evaluation_type=EvaluationType.REGRESSION,
                status=EvaluationStatus.SKIPPED,
                message="No baseline found. Run create_baseline() first."
            )
        
        if agent is None:
            from src.agent_core import CustomerServiceAgent
            agent = CustomerServiceAgent()
        
        regressions = []
        
        for name, baseline_data in baseline["queries"].items():
            query = baseline_data["query"]
            expected_length = baseline_data["response_length"]
            expected_duration = baseline_data["duration_ms"]
            
            # Run query
            query_start = time.time()
            response = agent.run(query)
            actual_duration = (time.time() - query_start) * 1000
            
            # Check for regressions
            if abs(len(response) - expected_length) > expected_length * 0.5:
                regressions.append({
                    "query_name": name,
                    "issue": "response_length_changed",
                    "expected": expected_length,
                    "actual": len(response)
                })
            
            if actual_duration > expected_duration + tolerance_ms:
                regressions.append({
                    "query_name": name,
                    "issue": "latency_increased",
                    "expected_ms": expected_duration,
                    "actual_ms": actual_duration
                })
        
        duration_ms = (time.time() - start_time) * 1000
        passed = len(regressions) == 0
        
        return EvaluationResult(
            test_name="regression_check",
            evaluation_type=EvaluationType.REGRESSION,
            status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
            message="No regressions detected" if passed else f"{len(regressions)} regressions found",
            duration_ms=duration_ms,
            details={
                "baseline_date": baseline.get("created_at"),
                "queries_checked": len(baseline["queries"]),
                "regressions": regressions
            }
        )


class EvaluationSuite:
    """
    Complete evaluation suite that combines all evaluators.
    """
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or os.path.join(os.path.dirname(__file__), "dataset.json")
        self.dataset = self._load_dataset()
        
        self.component_evaluator = ComponentEvaluator()
        self.system_evaluator = SystemEvaluator()
        self.performance_evaluator = PerformanceEvaluator()
        self.regression_evaluator = RegressionEvaluator()
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load evaluation dataset."""
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, 'r') as f:
                return json.load(f)
        return []
    
    def run_all_evaluations(
        self,
        agent: Any = None,
        include_performance: bool = True,
        include_regression: bool = False
    ) -> EvaluationReport:
        """
        Run all evaluations from the dataset.
        
        Args:
            agent: The agent instance
            include_performance: Whether to include performance tests
            include_regression: Whether to include regression tests
        
        Returns:
            Complete EvaluationReport
        """
        report = self.create_report()
        
        # Component-level evaluations
        for case in self.dataset:
            query = case["query"]
            expected_components = case.get("expected_components", {})
            
            # Tool selection test
            if "tools_called" in expected_components:
                result = self.component_evaluator.evaluate_tool_selection(
                    query=query,
                    expected_tools=expected_components["tools_called"],
                    agent=agent
                )
                report.add_result(result)
            
            # SQL generation test
            if "target_table" in expected_components:
                result = self.component_evaluator.evaluate_sql_generation(
                    query=query,
                    expected_table=expected_components["target_table"],
                    agent=agent
                )
                report.add_result(result)
        
        # System-level evaluations
        for case in self.dataset:
            query = case["query"]
            expected_answer = case.get("expected_final_answer")
            
            if expected_answer:
                result = self.system_evaluator.evaluate_end_to_end(
                    query=query,
                    expected_answer=expected_answer,
                    agent=agent
                )
                report.add_result(result)
        
        # Performance evaluations
        if include_performance:
            queries = [case["query"] for case in self.dataset]
            result = self.performance_evaluator.benchmark_queries(
                queries=queries,
                agent=agent
            )
            report.add_result(result)
        
        # Regression evaluations
        if include_regression:
            result = self.regression_evaluator.evaluate_regression(agent=agent)
            report.add_result(result)
        
        report.finalize()
        return report
    
    def create_report(self) -> EvaluationReport:
        """Create a new evaluation report."""
        return EvaluationReport(
            name="Complete Evaluation Suite",
            metadata={
                "dataset_path": self.dataset_path,
                "dataset_size": len(self.dataset)
            }
        )


def run_evaluation_suite(
    dataset_path: str = None,
    output_path: str = None,
    agent: Any = None
) -> EvaluationReport:
    """
    Convenience function to run the complete evaluation suite.
    
    Args:
        dataset_path: Path to evaluation dataset
        output_path: Path to save the report
        agent: The agent instance
    
    Returns:
        EvaluationReport
    """
    suite = EvaluationSuite(dataset_path)
    report = suite.run_all_evaluations(agent=agent)
    
    if output_path:
        report.save(output_path)
    
    report.print_summary()
    return report


if __name__ == "__main__":
    # Run evaluation suite when executed directly
    run_evaluation_suite(
        output_path=os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    )