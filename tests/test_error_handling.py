"""
Tests for the error_handling module.

This module tests error classification, retry logic, circuit breakers,
and error handling utilities.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from src.error_handling import (
    ErrorCategory, AgentError, LLMError, ToolExecutionError,
    DatabaseError, ConfigurationError, ValidationError,
    classify_error, wrap_error, RetryConfig,
    retry_on_error, CircuitBreaker, with_circuit_breaker,
    ErrorHandler, get_error_handler
)


class TestErrorCategory:
    """Tests for the ErrorCategory enum."""
    
    def test_error_categories_exist(self):
        """Test that all error categories are defined."""
        assert ErrorCategory.TRANSIENT.value == "transient"
        assert ErrorCategory.PERMANENT.value == "permanent"
        assert ErrorCategory.RATE_LIMIT.value == "rate_limit"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.AUTHENTICATION.value == "authentication"


class TestAgentError:
    """Tests for the AgentError class."""
    
    def test_basic_error(self):
        """Test creating a basic agent error."""
        error = AgentError("Something went wrong")
        
        assert error.message == "Something went wrong"
        assert error.category == ErrorCategory.PERMANENT
        assert error.original_error is None
        assert error.is_retryable is False
        assert error.retry_after is None
    
    def test_error_with_category(self):
        """Test creating an error with a specific category."""
        error = AgentError(
            "Connection failed",
            category=ErrorCategory.TRANSIENT
        )
        
        assert error.category == ErrorCategory.TRANSIENT
        assert error.is_retryable is True
    
    def test_error_with_original(self):
        """Test wrapping an original error."""
        original = ValueError("Original error")
        error = AgentError(
            "Wrapped error",
            category=ErrorCategory.TRANSIENT,
            original_error=original
        )
        
        assert error.original_error is original
    
    def test_error_with_retry_after(self):
        """Test error with retry_after value."""
        error = AgentError(
            "Rate limited",
            category=ErrorCategory.RATE_LIMIT,
            retry_after=30
        )
        
        assert error.retry_after == 30
        assert error.is_retryable is True
    
    def test_to_dict(self):
        """Test error serialization."""
        error = AgentError(
            "Test error",
            category=ErrorCategory.TIMEOUT,
            original_error=ValueError("original")
        )
        
        result = error.to_dict()
        
        assert result["error_type"] == "AgentError"
        assert result["message"] == "Test error"
        assert result["category"] == "timeout"
        assert result["is_retryable"] is True
        assert "original" in result["original_error"]


class TestErrorSubclasses:
    """Tests for AgentError subclasses."""
    
    def test_llm_error(self):
        """Test LLMError class."""
        error = LLMError("API error", category=ErrorCategory.RATE_LIMIT)
        
        assert isinstance(error, AgentError)
        assert error.message == "API error"
    
    def test_tool_execution_error(self):
        """Test ToolExecutionError class."""
        error = ToolExecutionError("Tool failed")
        
        assert isinstance(error, AgentError)
        assert error.message == "Tool failed"
    
    def test_database_error(self):
        """Test DatabaseError class."""
        error = DatabaseError("Connection failed")
        
        assert isinstance(error, AgentError)
    
    def test_configuration_error(self):
        """Test ConfigurationError class."""
        error = ConfigurationError("Missing API key")
        
        assert isinstance(error, AgentError)
    
    def test_validation_error(self):
        """Test ValidationError class."""
        error = ValidationError("Invalid input")
        
        assert isinstance(error, AgentError)


class TestClassifyError:
    """Tests for the classify_error function."""
    
    def test_classify_rate_limit(self):
        """Test classifying rate limit errors."""
        error = Exception("Rate limit exceeded")
        assert classify_error(error) == ErrorCategory.RATE_LIMIT
        
        error2 = Exception("429 Too Many Requests")
        assert classify_error(error2) == ErrorCategory.RATE_LIMIT
    
    def test_classify_timeout(self):
        """Test classifying timeout errors."""
        error = Exception("Request timeout")
        assert classify_error(error) == ErrorCategory.TIMEOUT
        
        error2 = Exception("Operation timed out")
        assert classify_error(error2) == ErrorCategory.TIMEOUT
    
    def test_classify_auth(self):
        """Test classifying authentication errors."""
        error = Exception("Unauthorized access")
        assert classify_error(error) == ErrorCategory.AUTHENTICATION
        
        error2 = Exception("403 Forbidden")
        assert classify_error(error2) == ErrorCategory.AUTHENTICATION
    
    def test_classify_validation(self):
        """Test classifying validation errors."""
        error = Exception("Invalid input")
        assert classify_error(error) == ErrorCategory.VALIDATION
        
        error2 = Exception("400 Bad Request")
        assert classify_error(error2) == ErrorCategory.VALIDATION
    
    def test_classify_transient(self):
        """Test classifying transient errors."""
        error = Exception("Connection reset")
        assert classify_error(error) == ErrorCategory.TRANSIENT
        
        error2 = Exception("Service temporarily unavailable")
        assert classify_error(error2) == ErrorCategory.TRANSIENT
    
    def test_classify_permanent_default(self):
        """Test that unknown errors default to permanent."""
        error = Exception("Some unknown error")
        assert classify_error(error) == ErrorCategory.PERMANENT


class TestWrapError:
    """Tests for the wrap_error function."""
    
    def test_wrap_generic_error(self):
        """Test wrapping a generic exception."""
        original = ValueError("Something went wrong")
        wrapped = wrap_error(original)
        
        assert isinstance(wrapped, AgentError)
        assert wrapped.original_error is original
        assert "Something went wrong" in wrapped.message
    
    def test_wrap_with_custom_class(self):
        """Test wrapping with a custom error class."""
        original = Exception("API failed")
        wrapped = wrap_error(original, wrapper_class=LLMError)
        
        assert isinstance(wrapped, LLMError)
    
    def test_wrap_with_custom_message(self):
        """Test wrapping with a custom message."""
        original = ValueError("Original")
        wrapped = wrap_error(original, message="Custom message")
        
        assert wrapped.message == "Custom message"


class TestRetryConfig:
    """Tests for the RetryConfig class."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
    
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            jitter=False
        )
        
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.jitter is False
    
    def test_calculate_delay(self):
        """Test delay calculation with exponential backoff."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False
        )
        
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0
    
    def test_calculate_delay_with_max(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=10.0,
            max_delay=50.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # 10 * 2^3 = 80, but should be capped at 50
        assert config.calculate_delay(3) == 50.0
    
    def test_calculate_delay_with_jitter(self):
        """Test that jitter is added when enabled."""
        config = RetryConfig(
            base_delay=10.0,
            jitter=True
        )
        
        # With jitter, delay should be >= base_delay
        # and <= base_delay * 1.25
        delay = config.calculate_delay(0)
        assert delay >= 10.0
        assert delay <= 12.5
    
    def test_should_retry(self):
        """Test should_retry method."""
        config = RetryConfig()
        
        # Retryable categories
        assert config.should_retry(AgentError("test", category=ErrorCategory.TRANSIENT))
        assert config.should_retry(AgentError("test", category=ErrorCategory.RATE_LIMIT))
        assert config.should_retry(AgentError("test", category=ErrorCategory.TIMEOUT))
        
        # Non-retryable categories
        assert not config.should_retry(AgentError("test", category=ErrorCategory.PERMANENT))
        assert not config.should_retry(AgentError("test", category=ErrorCategory.VALIDATION))
        assert not config.should_retry(AgentError("test", category=ErrorCategory.AUTHENTICATION))


class TestRetryOnError:
    """Tests for the retry_on_error decorator."""
    
    def test_successful_call(self):
        """Test that successful calls pass through."""
        call_count = 0
        
        @retry_on_error()
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_transient_error(self):
        """Test retry on transient errors."""
        call_count = 0
        
        @retry_on_error(RetryConfig(max_retries=3, base_delay=0.01, jitter=False))
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AgentError("Transient", category=ErrorCategory.TRANSIENT)
            return "success"
        
        result = failing_then_success()
        
        assert result == "success"
        assert call_count == 3
    
    def test_no_retry_on_permanent_error(self):
        """Test no retry on permanent errors."""
        call_count = 0
        
        @retry_on_error()
        def permanent_failure():
            nonlocal call_count
            call_count += 1
            raise AgentError("Permanent", category=ErrorCategory.PERMANENT)
        
        with pytest.raises(AgentError, match="Permanent"):
            permanent_failure()
        
        assert call_count == 1
    
    def test_max_retries_exceeded(self):
        """Test that max retries is respected."""
        call_count = 0
        
        @retry_on_error(RetryConfig(max_retries=2, base_delay=0.01))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise AgentError("Always fails", category=ErrorCategory.TRANSIENT)
        
        with pytest.raises(AgentError, match="Always fails"):
            always_fail()
        
        # Initial call + 2 retries = 3 total calls
        assert call_count == 3
    
    def test_retry_callback(self):
        """Test retry callback is called."""
        retries = []
        
        def on_retry(error, attempt, delay):
            retries.append((attempt, delay))
        
        @retry_on_error(
            RetryConfig(max_retries=2, base_delay=0.01, jitter=False),
            on_retry=on_retry
        )
        def failing_then_success():
            if len(retries) < 2:
                raise AgentError("Retry me", category=ErrorCategory.TRANSIENT)
            return "success"
        
        result = failing_then_success()
        
        assert result == "success"
        assert len(retries) == 2


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""
    
    def test_initial_state_closed(self):
        """Test that circuit breaker starts closed."""
        cb = CircuitBreaker()
        
        assert cb.state == CircuitBreaker.State.CLOSED
        assert cb.can_execute() is True
    
    def test_opens_after_threshold(self):
        """Test that circuit opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Record failures
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == CircuitBreaker.State.OPEN
        assert cb.can_execute() is False
    
    def test_closes_on_success(self):
        """Test that circuit closes on success."""
        cb = CircuitBreaker(failure_threshold=2)
        
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.State.OPEN
        
        # Simulate recovery timeout
        cb._last_failure_time = time.time() - 60
        assert cb.state == CircuitBreaker.State.HALF_OPEN
        
        cb.record_success()
        assert cb.state == CircuitBreaker.State.CLOSED
    
    def test_half_open_allows_limited_calls(self):
        """Test that half-open state allows limited calls."""
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_calls=2
        )
        
        cb.record_failure()
        assert cb.state == CircuitBreaker.State.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        assert cb.state == CircuitBreaker.State.HALF_OPEN
        assert cb.can_execute() is True
        assert cb.can_execute() is True
        assert cb.can_execute() is False  # Exceeded half_open_max_calls
    
    def test_reset(self):
        """Test resetting circuit breaker."""
        cb = CircuitBreaker(failure_threshold=1)
        
        cb.record_failure()
        assert cb.state == CircuitBreaker.State.OPEN
        
        cb.reset()
        
        assert cb.state == CircuitBreaker.State.CLOSED
        assert cb._failure_count == 0
    
    def test_to_dict(self):
        """Test circuit breaker serialization."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        
        result = cb.to_dict()
        
        assert result["state"] == "closed"
        assert result["failure_threshold"] == 5
        assert result["recovery_timeout"] == 30.0


class TestWithCircuitBreaker:
    """Tests for the with_circuit_breaker decorator."""
    
    def test_successful_call(self):
        """Test successful call through circuit breaker."""
        cb = CircuitBreaker()
        
        @with_circuit_breaker(cb)
        def successful():
            return "success"
        
        result = successful()
        
        assert result == "success"
        assert cb.state == CircuitBreaker.State.CLOSED
    
    def test_failure_recorded(self):
        """Test that failures are recorded."""
        cb = CircuitBreaker(failure_threshold=1)
        
        @with_circuit_breaker(cb)
        def failing():
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            failing()
        
        assert cb._failure_count == 1
    
    def test_open_circuit_blocks_calls(self):
        """Test that open circuit blocks calls."""
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()  # Open the circuit
        
        @with_circuit_breaker(cb)
        def blocked():
            return "should not reach"
        
        with pytest.raises(AgentError, match="Circuit breaker is open"):
            blocked()
    
    def test_fallback_called_when_open(self):
        """Test that fallback is called when circuit is open."""
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        
        def fallback_fn():
            return "fallback result"
        
        @with_circuit_breaker(cb, fallback=fallback_fn)
        def with_fallback():
            return "primary result"
        
        result = with_fallback()
        
        assert result == "fallback result"


class TestErrorHandler:
    """Tests for the ErrorHandler class."""
    
    def test_handle_agent_error(self):
        """Test handling an AgentError."""
        handler = ErrorHandler()
        error = AgentError("Test error", category=ErrorCategory.TRANSIENT)
        
        result = handler.handle_error(error)
        
        assert result is error
    
    def test_handle_generic_error(self):
        """Test handling a generic exception."""
        handler = ErrorHandler()
        error = ValueError("Something went wrong")
        
        result = handler.handle_error(error)
        
        assert isinstance(result, AgentError)
        assert result.original_error is error
    
    def test_handle_error_with_context(self):
        """Test handling error with context."""
        handler = ErrorHandler(logger=MagicMock())
        error = ValueError("Test error")
        
        result = handler.handle_error(error, context={"query": "SELECT 1"})
        
        assert isinstance(result, AgentError)
    
    def test_get_circuit_breaker(self):
        """Test getting circuit breaker."""
        handler = ErrorHandler()
        
        cb = handler.get_circuit_breaker("llm", failure_threshold=10)
        
        assert isinstance(cb, CircuitBreaker)
        assert cb.failure_threshold == 10
        
        # Getting same name returns same instance
        cb2 = handler.get_circuit_breaker("llm")
        assert cb is cb2
    
    def test_get_all_circuit_breaker_states(self):
        """Test getting all circuit breaker states."""
        handler = ErrorHandler()
        
        handler.get_circuit_breaker("llm")
        handler.get_circuit_breaker("database")
        
        states = handler.get_all_circuit_breaker_states()
        
        assert "llm" in states
        assert "database" in states


class TestGetErrorHandler:
    """Tests for the get_error_handler function."""
    
    def test_returns_singleton(self):
        """Test that get_error_handler returns a singleton."""
        # Reset the global instance
        import src.error_handling as eh
        eh._error_handler = None
        
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        assert handler1 is handler2
    
    def test_with_logger(self):
        """Test getting handler with logger."""
        import src.error_handling as eh
        eh._error_handler = None
        
        logger = MagicMock()
        handler = get_error_handler(logger=logger)
        
        assert handler.logger is logger