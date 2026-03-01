"""
Tests for the evaluation framework module.

This module tests the comprehensive evaluation capabilities including
component-level, system-level, performance, and regression evaluations.
"""

import pytest
import json
import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from evals.evaluation_framework import (
    EvaluationStatus, EvaluationType, EvaluationResult, EvaluationReport,
    Evaluator, ComponentEvaluator, SystemEvaluator, PerformanceEvaluator,
    RegressionEvaluator, EvaluationSuite, run_evaluation_suite
)


class TestEvaluationStatus:
    """Tests for EvaluationStatus enum."""
    
    def test_status_values(self):
        """Test that all expected statuses exist."""
        assert EvaluationStatus.PASSED.value == "passed"
        assert EvaluationStatus.FAILED.value == "failed"
        assert EvaluationStatus.ERROR.value == "error"
        assert EvaluationStatus.SKIPPED.value == "skipped"


class TestEvaluationType:
    """Tests for EvaluationType enum."""
    
    def test_type_values(self):
        """Test that all expected types exist."""
        assert EvaluationType.COMPONENT.value == "component"
        assert EvaluationType.SYSTEM.value == "system"
        assert EvaluationType.PERFORMANCE.value == "performance"
        assert EvaluationType.REGRESSION.value == "regression"


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            test_name="test_1",
            evaluation_type=EvaluationType.COMPONENT,
            status=EvaluationStatus.PASSED,
            message="Test passed"
        )
        
        assert result.test_name == "test_1"
        assert result.evaluation_type == EvaluationType.COMPONENT
        assert result.status == EvaluationStatus.PASSED
        assert result.message == "Test passed"
        assert result.duration_ms == 0.0
        assert result.details == {}
        assert result.error is None
        assert result.timestamp is not None
    
    def test_result_with_details(self):
        """Test result with additional details."""
        result = EvaluationResult(
            test_name="test_2",
            evaluation_type=EvaluationType.SYSTEM,
            status=EvaluationStatus.FAILED,
            message="Test failed",
            duration_ms=150.5,
            details={"key": "value"},
            error=ValueError("test error")
        )
        
        assert result.duration_ms == 150.5
        assert result.details["key"] == "value"
        assert result.error is not None
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = EvaluationResult(
            test_name="test_3",
            evaluation_type=EvaluationType.PERFORMANCE,
            status=EvaluationStatus.PASSED,
            message="Performance OK",
            duration_ms=100.0,
            details={"latency": 50}
        )
        
        d = result.to_dict()
        
        assert d["test_name"] == "test_3"
        assert d["evaluation_type"] == "performance"
        assert d["status"] == "passed"
        assert d["message"] == "Performance OK"
        assert d["duration_ms"] == 100.0
        assert d["details"]["latency"] == 50
        assert d["error"] is None


class TestEvaluationReport:
    """Tests for EvaluationReport dataclass."""
    
    def test_report_creation(self):
        """Test creating an evaluation report."""
        report = EvaluationReport(name="Test Report")
        
        assert report.name == "Test Report"
        assert report.results == []
        assert report.start_time is not None
        assert report.end_time is None
        assert report.metadata == {}
    
    def test_report_add_result(self):
        """Test adding results to report."""
        report = EvaluationReport(name="Test Report")
        
        result1 = EvaluationResult(
            test_name="test_1",
            evaluation_type=EvaluationType.COMPONENT,
            status=EvaluationStatus.PASSED
        )
        result2 = EvaluationResult(
            test_name="test_2",
            evaluation_type=EvaluationType.SYSTEM,
            status=EvaluationStatus.FAILED
        )
        
        report.add_result(result1)
        report.add_result(result2)
        
        assert len(report.results) == 2
        assert report.total == 2
        assert report.passed == 1
        assert report.failed == 1
    
    def test_report_summary_counts(self):
        """Test report summary counts."""
        report = EvaluationReport(name="Test Report")
        
        statuses = [
            EvaluationStatus.PASSED,
            EvaluationStatus.PASSED,
            EvaluationStatus.FAILED,
            EvaluationStatus.ERROR,
            EvaluationStatus.SKIPPED
        ]
        
        for i, status in enumerate(statuses):
            report.add_result(EvaluationResult(
                test_name=f"test_{i}",
                evaluation_type=EvaluationType.COMPONENT,
                status=status
            ))
        
        assert report.total == 5
        assert report.passed == 2
        assert report.failed == 1
        assert report.errors == 1
        assert report.skipped == 1
    
    def test_report_pass_rate(self):
        """Test pass rate calculation."""
        report = EvaluationReport(name="Test Report")
        
        # Empty report
        assert report.pass_rate == 0.0
        
        # Add results
        for i in range(10):
            status = EvaluationStatus.PASSED if i < 7 else EvaluationStatus.FAILED
            report.add_result(EvaluationResult(
                test_name=f"test_{i}",
                evaluation_type=EvaluationType.COMPONENT,
                status=status
            ))
        
        assert report.pass_rate == 70.0
    
    def test_report_finalize(self):
        """Test finalizing a report."""
        report = EvaluationReport(name="Test Report")
        
        assert report.end_time is None
        
        report.finalize()
        
        assert report.end_time is not None
    
    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        report = EvaluationReport(name="Test Report", metadata={"version": "1.0"})
        report.add_result(EvaluationResult(
            test_name="test_1",
            evaluation_type=EvaluationType.COMPONENT,
            status=EvaluationStatus.PASSED
        ))
        report.finalize()
        
        d = report.to_dict()
        
        assert d["name"] == "Test Report"
        assert d["metadata"]["version"] == "1.0"
        assert d["summary"]["total"] == 1
        assert d["summary"]["passed"] == 1
        assert d["summary"]["pass_rate"] == 100.0
        assert len(d["results"]) == 1
    
    def test_report_save(self):
        """Test saving report to file."""
        report = EvaluationReport(name="Test Report")
        report.add_result(EvaluationResult(
            test_name="test_1",
            evaluation_type=EvaluationType.COMPONENT,
            status=EvaluationStatus.PASSED
        ))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            report.save(temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert data["name"] == "Test Report"
            assert len(data["results"]) == 1
        finally:
            os.unlink(temp_path)


class TestEvaluator:
    """Tests for the base Evaluator class."""
    
    def test_evaluator_creation(self):
        """Test creating an evaluator."""
        evaluator = Evaluator("Test Evaluator")
        
        assert evaluator.name == "Test Evaluator"
        assert evaluator.error_handler is not None
    
    def test_evaluator_create_report(self):
        """Test creating a report from evaluator."""
        evaluator = Evaluator("Test Evaluator")
        
        report = evaluator.create_report(metadata={"key": "value"})
        
        assert report.name == "Test Evaluator"
        assert report.metadata["key"] == "value"
    
    def test_evaluator_evaluate_bool_result(self):
        """Test evaluating a function that returns bool."""
        evaluator = Evaluator("Test Evaluator")
        
        result = evaluator.evaluate(
            test_name="bool_test",
            evaluation_type=EvaluationType.COMPONENT,
            test_func=lambda: True
        )
        
        assert result.status == EvaluationStatus.PASSED
        
        result = evaluator.evaluate(
            test_name="bool_test_fail",
            evaluation_type=EvaluationType.COMPONENT,
            test_func=lambda: False
        )
        
        assert result.status == EvaluationStatus.FAILED
    
    def test_evaluator_evaluate_dict_result(self):
        """Test evaluating a function that returns dict."""
        evaluator = Evaluator("Test Evaluator")
        
        result = evaluator.evaluate(
            test_name="dict_test",
            evaluation_type=EvaluationType.SYSTEM,
            test_func=lambda: {"passed": True, "message": "OK"}
        )
        
        assert result.status == EvaluationStatus.PASSED
        assert result.message == "OK"
    
    def test_evaluator_evaluate_exception(self):
        """Test evaluating a function that raises exception."""
        evaluator = Evaluator("Test Evaluator")
        
        def failing_func():
            raise ValueError("Test error")
        
        result = evaluator.evaluate(
            test_name="error_test",
            evaluation_type=EvaluationType.COMPONENT,
            test_func=failing_func
        )
        
        assert result.status == EvaluationStatus.ERROR
        assert "Test error" in result.message
        assert result.error is not None


class TestComponentEvaluator:
    """Tests for ComponentEvaluator class."""
    
    def test_evaluator_creation(self):
        """Test creating component evaluator."""
        evaluator = ComponentEvaluator()
        
        assert evaluator.name == "Component Evaluation"
    
    def test_evaluate_tool_selection_missing_tools(self):
        """Test tool selection evaluation with missing tools."""
        evaluator = ComponentEvaluator()
        
        # Create mock agent
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.tool_calls = [{"name": "tool_a"}]
        mock_agent.llm_with_tools.invoke.return_value = mock_response
        mock_agent._get_system_prompt.return_value = "System prompt"
        
        result = evaluator.evaluate_tool_selection(
            query="Test query",
            expected_tools=["tool_a", "tool_b"],
            agent=mock_agent
        )
        
        assert result.status == EvaluationStatus.FAILED
        assert "tool_b" in result.details["missing_tools"]
    
    def test_evaluate_tool_selection_all_tools_present(self):
        """Test tool selection with all expected tools."""
        evaluator = ComponentEvaluator()
        
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.tool_calls = [
            {"name": "tool_a"},
            {"name": "tool_b"}
        ]
        mock_agent.llm_with_tools.invoke.return_value = mock_response
        mock_agent._get_system_prompt.return_value = "System prompt"
        
        result = evaluator.evaluate_tool_selection(
            query="Test query",
            expected_tools=["tool_a", "tool_b"],
            agent=mock_agent
        )
        
        assert result.status == EvaluationStatus.PASSED
        assert result.details["missing_tools"] == []


class TestSystemEvaluator:
    """Tests for SystemEvaluator class."""
    
    def test_evaluator_creation(self):
        """Test creating system evaluator."""
        evaluator = SystemEvaluator()
        
        assert evaluator.name == "System Evaluation"
    
    def test_evaluate_end_to_end_passed(self):
        """Test end-to-end evaluation that passes."""
        evaluator = SystemEvaluator()
        
        mock_agent = Mock()
        mock_agent.run.return_value = "The answer is 42 units in stock."
        
        result = evaluator.evaluate_end_to_end(
            query="How many units?",
            expected_answer="42",
            agent=mock_agent
        )
        
        assert result.status == EvaluationStatus.PASSED
        assert "42" in result.message
    
    def test_evaluate_end_to_end_failed(self):
        """Test end-to-end evaluation that fails."""
        evaluator = SystemEvaluator()
        
        mock_agent = Mock()
        mock_agent.run.return_value = "The answer is 100 units."
        
        result = evaluator.evaluate_end_to_end(
            query="How many units?",
            expected_answer="42",
            agent=mock_agent
        )
        
        assert result.status == EvaluationStatus.FAILED
    
    def test_evaluate_response_quality_passed(self):
        """Test response quality evaluation that passes."""
        evaluator = SystemEvaluator()
        
        result = evaluator.evaluate_response_quality(
            query="Test query",
            response="This is a good response with enough content.",
            criteria={"min_length": 10}
        )
        
        assert result.status == EvaluationStatus.PASSED
    
    def test_evaluate_response_quality_failed(self):
        """Test response quality evaluation that fails."""
        evaluator = SystemEvaluator()
        
        result = evaluator.evaluate_response_quality(
            query="Test query",
            response="Short",
            criteria={"min_length": 10}
        )
        
        assert result.status == EvaluationStatus.FAILED
        assert "too short" in result.message.lower()


class TestPerformanceEvaluator:
    """Tests for PerformanceEvaluator class."""
    
    def test_evaluator_creation(self):
        """Test creating performance evaluator."""
        evaluator = PerformanceEvaluator()
        
        assert evaluator.name == "Performance Evaluation"
    
    def test_evaluate_latency_passed(self):
        """Test latency evaluation that passes."""
        evaluator = PerformanceEvaluator()
        
        mock_agent = Mock()
        mock_agent.run.return_value = "Response"
        
        result = evaluator.evaluate_latency(
            query="Test query",
            max_latency_ms=10000,  # 10 seconds
            agent=mock_agent
        )
        
        assert result.status == EvaluationStatus.PASSED
        assert result.duration_ms >= 0  # Mock is instant, so duration can be 0
    
    def test_benchmark_queries(self):
        """Test benchmarking multiple queries."""
        evaluator = PerformanceEvaluator()
        
        mock_agent = Mock()
        mock_agent.run.return_value = "Response"
        
        result = evaluator.benchmark_queries(
            queries=["Query 1", "Query 2"],
            agent=mock_agent,
            warmup_runs=0
        )
        
        assert result.status == EvaluationStatus.PASSED
        assert result.details["query_count"] == 2
        assert "avg_latency_ms" in result.details
        assert len(result.details["latencies"]) == 2


class TestRegressionEvaluator:
    """Tests for RegressionEvaluator class."""
    
    def test_evaluator_creation(self):
        """Test creating regression evaluator."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_path = os.path.join(temp_dir, "baseline.json")
            evaluator = RegressionEvaluator(baseline_path=baseline_path)
            
            assert evaluator.name == "Regression Evaluation"
    
    def test_evaluate_regression_no_baseline(self):
        """Test regression evaluation without baseline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_path = os.path.join(temp_dir, "nonexistent_baseline.json")
            evaluator = RegressionEvaluator(baseline_path=baseline_path)
            
            result = evaluator.evaluate_regression()
            
            assert result.status == EvaluationStatus.SKIPPED
            assert "No baseline found" in result.message
    
    def test_create_and_load_baseline(self):
        """Test creating and loading baseline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_path = os.path.join(temp_dir, "baseline.json")
            evaluator = RegressionEvaluator(baseline_path=baseline_path)
            
            mock_agent = Mock()
            mock_agent.run.return_value = "Test response"
            
            baseline = evaluator.create_baseline(
                queries=[
                    {"name": "query_1", "query": "Test query 1"}
                ],
                agent=mock_agent
            )
            
            assert "created_at" in baseline
            assert "query_1" in baseline["queries"]
            
            # Load baseline
            loaded = evaluator.load_baseline()
            
            assert loaded is not None
            assert "query_1" in loaded["queries"]


class TestEvaluationSuite:
    """Tests for EvaluationSuite class."""
    
    def test_suite_creation(self):
        """Test creating evaluation suite."""
        suite = EvaluationSuite()
        
        assert suite.component_evaluator is not None
        assert suite.system_evaluator is not None
        assert suite.performance_evaluator is not None
        assert suite.regression_evaluator is not None
    
    def test_suite_create_report(self):
        """Test creating report from suite."""
        suite = EvaluationSuite()
        
        report = suite.create_report()
        
        assert report.name == "Complete Evaluation Suite"
    
    def test_suite_with_custom_dataset(self):
        """Test suite with custom dataset path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "dataset.json")
            
            # Create test dataset
            dataset = [
                {
                    "query": "Test query",
                    "expected_components": {
                        "tools_called": ["tool_a"]
                    },
                    "expected_final_answer": "42"
                }
            ]
            
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f)
            
            suite = EvaluationSuite(dataset_path=dataset_path)
            
            assert len(suite.dataset) == 1
            assert suite.dataset[0]["query"] == "Test query"


class TestRunEvaluationSuite:
    """Tests for run_evaluation_suite convenience function."""
    
    def test_run_evaluation_suite_basic(self):
        """Test running evaluation suite."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "dataset.json")
            output_path = os.path.join(temp_dir, "report.json")
            
            # Create minimal dataset
            dataset = [
                {
                    "query": "Test query",
                    "expected_final_answer": "42"
                }
            ]
            
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f)
            
            # Mock agent
            mock_agent = Mock()
            mock_agent.run.return_value = "The answer is 42"
            mock_response = Mock()
            mock_response.tool_calls = []
            mock_agent.llm_with_tools.invoke.return_value = mock_response
            mock_agent._get_system_prompt.return_value = "System"
            
            report = run_evaluation_suite(
                dataset_path=dataset_path,
                output_path=output_path,
                agent=mock_agent
            )
            
            assert report is not None
            assert os.path.exists(output_path)