"""
error_handling.py

This module provides comprehensive error handling capabilities for the Agentic Workflow
application. It implements error classification, retry logic with exponential backoff,
and graceful degradation patterns to improve system resilience.

The error handling system categorizes errors into transient (retryable) and permanent
(non-retryable) types, and provides utilities for consistent error handling across
the application.
"""

import time
import random
from typing import Optional, Callable, Any, Dict, List, Type
from functools import wraps
from enum import Enum


class ErrorCategory(Enum):
    """
    Enumeration of error categories for classification.
    
    Categories:
        TRANSIENT: Temporary errors that may succeed on retry.
        PERMANENT: Errors that will not succeed regardless of retries.
        RATE_LIMIT: Rate limiting errors that require waiting before retry.
        TIMEOUT: Timeout errors that may succeed on retry.
        VALIDATION: Input validation errors that are permanent.
        AUTHENTICATION: Authentication/authorization errors that are permanent.
    """
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"


class AgentError(Exception):
    """
    Base exception class for all agent-related errors.
    
    This class provides structured error information including category,
    original error, and whether the error is retryable.
    
    Attributes:
        message (str): Human-readable error message.
        category (ErrorCategory): The category of this error.
        original_error (Optional[Exception]): The underlying exception.
        is_retryable (bool): Whether this error can be retried.
        retry_after (Optional[int]): Seconds to wait before retry (for rate limits).
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.PERMANENT,
        original_error: Optional[Exception] = None,
        retry_after: Optional[int] = None
    ):
        """
        Initialize the agent error.
        
        Args:
            message (str): Human-readable error message.
            category (ErrorCategory): The category of this error.
            original_error (Optional[Exception]): The underlying exception.
            retry_after (Optional[int]): Seconds to wait before retry.
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.original_error = original_error
        self.is_retryable = category in (
            ErrorCategory.TRANSIENT,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.TIMEOUT
        )
        self.retry_after = retry_after
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary for logging and serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the error.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "is_retryable": self.is_retryable,
            "retry_after": self.retry_after,
            "original_error": str(self.original_error) if self.original_error else None
        }


class LLMError(AgentError):
    """
    Exception for LLM-related errors.
    
    Raised when there are issues with LLM API calls including rate limits,
    timeouts, and API errors.
    """
    pass


class ToolExecutionError(AgentError):
    """
    Exception for tool execution errors.
    
    Raised when a tool fails to execute properly, such as SQL syntax errors
    or database connection issues.
    """
    pass


class DatabaseError(AgentError):
    """
    Exception for database-related errors.
    
    Raised when there are issues with database connectivity or operations.
    """
    pass


class ConfigurationError(AgentError):
    """
    Exception for configuration-related errors.
    
    Raised when there are issues with application configuration,
    such as missing API keys or invalid settings.
    """
    pass


class ValidationError(AgentError):
    """
    Exception for input validation errors.
    
    Raised when user input or tool arguments fail validation.
    """
    pass


# Error classification rules
# Maps exception types or error message patterns to error categories
TRANSIENT_PATTERNS = [
    "connection reset",
    "connection refused",
    "temporarily unavailable",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "gateway timeout",
    "network error",
    "socket error",
]

RATE_LIMIT_PATTERNS = [
    "rate limit",
    "too many requests",
    "quota exceeded",
    "requests per minute",
    "requests per second",
    "429",
]

TIMEOUT_PATTERNS = [
    "timeout",
    "timed out",
    "deadline exceeded",
    "request timeout",
]

AUTH_PATTERNS = [
    "unauthorized",
    "forbidden",
    "invalid api key",
    "authentication failed",
    "access denied",
    "401",
    "403",
]

VALIDATION_PATTERNS = [
    "invalid",
    "validation error",
    "malformed",
    "syntax error",
    "bad request",
    "400",
]


def classify_error(error: Exception) -> ErrorCategory:
    """
    Classify an error into a category based on its type and message.
    
    Args:
        error (Exception): The error to classify.
        
    Returns:
        ErrorCategory: The classified error category.
    """
    error_message = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Check for rate limit patterns
    for pattern in RATE_LIMIT_PATTERNS:
        if pattern in error_message:
            return ErrorCategory.RATE_LIMIT
    
    # Check for timeout patterns
    for pattern in TIMEOUT_PATTERNS:
        if pattern in error_message:
            return ErrorCategory.TIMEOUT
    
    # Check for authentication patterns
    for pattern in AUTH_PATTERNS:
        if pattern in error_message:
            return ErrorCategory.AUTHENTICATION
    
    # Check for validation patterns
    for pattern in VALIDATION_PATTERNS:
        if pattern in error_message:
            return ErrorCategory.VALIDATION
    
    # Check for transient patterns
    for pattern in TRANSIENT_PATTERNS:
        if pattern in error_message:
            return ErrorCategory.TRANSIENT
    
    # Check specific exception types
    if "timeout" in error_type:
        return ErrorCategory.TIMEOUT
    
    if "connection" in error_type:
        return ErrorCategory.TRANSIENT
    
    # Default to permanent
    return ErrorCategory.PERMANENT


def wrap_error(
    error: Exception,
    wrapper_class: Type[AgentError] = AgentError,
    message: Optional[str] = None
) -> AgentError:
    """
    Wrap a generic exception in an AgentError with classification.
    
    Args:
        error (Exception): The original error to wrap.
        wrapper_class (Type[AgentError]): The AgentError subclass to use.
        message (Optional[str]): Custom error message.
        
    Returns:
        AgentError: The wrapped error with classification.
    """
    category = classify_error(error)
    return wrapper_class(
        message=message or str(error),
        category=category,
        original_error=error
    )


class RetryConfig:
    """
    Configuration for retry behavior.
    
    Attributes:
        max_retries (int): Maximum number of retry attempts.
        base_delay (float): Base delay in seconds between retries.
        max_delay (float): Maximum delay in seconds between retries.
        exponential_base (float): Base for exponential backoff calculation.
        jitter (bool): Whether to add random jitter to delays.
        retryable_categories (List[ErrorCategory]): Error categories to retry.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_categories: Optional[List[ErrorCategory]] = None
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries (int): Maximum number of retry attempts.
            base_delay (float): Base delay in seconds between retries.
            max_delay (float): Maximum delay in seconds between retries.
            exponential_base (float): Base for exponential backoff calculation.
            jitter (bool): Whether to add random jitter to delays.
            retryable_categories (Optional[List[ErrorCategory]]): Categories to retry.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_categories = retryable_categories or [
            ErrorCategory.TRANSIENT,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.TIMEOUT
        ]
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate the delay for a given retry attempt.
        
        Uses exponential backoff with optional jitter.
        
        Args:
            attempt (int): The retry attempt number (0-indexed).
            
        Returns:
            float: The delay in seconds.
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter between 0% and 25% of the delay
            jitter_amount = delay * random.uniform(0, 0.25)
            delay += jitter_amount
        
        return delay
    
    def should_retry(self, error: AgentError) -> bool:
        """
        Determine if an error should be retried.
        
        Args:
            error (AgentError): The error to check.
            
        Returns:
            bool: True if the error should be retried.
        """
        return error.category in self.retryable_categories


# Default retry configurations
DEFAULT_RETRY_CONFIG = RetryConfig()
LLM_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=2.0,
    max_delay=30.0,
    retryable_categories=[
        ErrorCategory.TRANSIENT,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.TIMEOUT
    ]
)
DATABASE_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    base_delay=0.5,
    max_delay=5.0,
    retryable_categories=[
        ErrorCategory.TRANSIENT,
        ErrorCategory.TIMEOUT
    ]
)


def retry_on_error(
    retry_config: Optional[RetryConfig] = None,
    error_wrapper: Type[AgentError] = AgentError,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
) -> Callable:
    """
    Decorator to retry a function on retryable errors.
    
    This decorator wraps a function to automatically retry it when
    retryable errors occur, using exponential backoff.
    
    Args:
        retry_config (Optional[RetryConfig]): Retry configuration to use.
        error_wrapper (Type[AgentError]): Error class to wrap exceptions in.
        on_retry (Optional[Callable]): Callback for retry events.
        
    Returns:
        Callable: The decorated function with retry logic.
    """
    config = retry_config or DEFAULT_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except AgentError as e:
                    last_error = e
                    if not config.should_retry(e) or attempt >= config.max_retries:
                        raise
                except Exception as e:
                    last_error = wrap_error(e, error_wrapper)
                    if not config.should_retry(last_error) or attempt >= config.max_retries:
                        raise last_error
                
                # Calculate delay and wait
                delay = config.calculate_delay(attempt)
                
                # Override with retry_after if specified (for rate limits)
                if isinstance(last_error, AgentError) and last_error.retry_after:
                    delay = max(delay, last_error.retry_after)
                
                # Call retry callback if provided
                if on_retry:
                    on_retry(last_error, attempt + 1, delay)
                
                time.sleep(delay)
            
            # Should not reach here, but raise last error just in case
            raise last_error
        
        return wrapper
    
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    
    The circuit breaker prevents repeated calls to a failing service,
    allowing it time to recover. It has three states:
    - CLOSED: Normal operation, calls pass through.
    - OPEN: Calls are blocked, service is considered down.
    - HALF_OPEN: Testing if service has recovered.
    
    Attributes:
        failure_threshold (int): Number of failures before opening.
        recovery_timeout (float): Seconds to wait before attempting recovery.
        half_open_max_calls (int): Max calls allowed in half-open state.
    """
    
    class State(Enum):
        """Circuit breaker states."""
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold (int): Failures before opening circuit.
            recovery_timeout (float): Seconds before attempting recovery.
            half_open_max_calls (int): Max calls in half-open state.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._state = self.State.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
    
    @property
    def state(self) -> State:
        """
        Get the current state of the circuit breaker.
        
        Returns:
            State: The current state.
        """
        if self._state == self.State.OPEN:
            # Check if we should transition to half-open
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = self.State.HALF_OPEN
                    self._half_open_calls = 0
        
        return self._state
    
    def can_execute(self) -> bool:
        """
        Check if execution is allowed.
        
        Returns:
            bool: True if execution is allowed.
        """
        state = self.state
        
        if state == self.State.CLOSED:
            return True
        
        if state == self.State.HALF_OPEN:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        
        return False  # OPEN state
    
    def record_success(self) -> None:
        """
        Record a successful execution.
        
        Resets the failure count and closes the circuit if in half-open state.
        """
        if self._state == self.State.HALF_OPEN:
            self._state = self.State.CLOSED
        self._failure_count = 0
    
    def record_failure(self) -> None:
        """
        Record a failed execution.
        
        Increments failure count and may open the circuit.
        """
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == self.State.HALF_OPEN:
            self._state = self.State.OPEN
        elif self._failure_count >= self.failure_threshold:
            self._state = self.State.OPEN
    
    def reset(self) -> None:
        """
        Reset the circuit breaker to closed state.
        """
        self._state = self.State.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the circuit breaker state to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation.
        """
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": self._last_failure_time
        }


def with_circuit_breaker(
    circuit_breaker: CircuitBreaker,
    fallback: Optional[Callable[[], Any]] = None
) -> Callable:
    """
    Decorator to wrap a function with circuit breaker protection.
    
    Args:
        circuit_breaker (CircuitBreaker): The circuit breaker to use.
        fallback (Optional[Callable]): Fallback function to call when circuit is open.
        
    Returns:
        Callable: The decorated function with circuit breaker protection.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not circuit_breaker.can_execute():
                if fallback:
                    return fallback()
                raise AgentError(
                    message="Circuit breaker is open",
                    category=ErrorCategory.TRANSIENT
                )
            
            try:
                result = func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure()
                raise
        
        return wrapper
    
    return decorator


class ErrorHandler:
    """
    Centralized error handler for consistent error processing.
    
    This class provides a unified interface for handling errors across
    the application, including logging, classification, and recovery.
    
    Attributes:
        logger: Logger instance for error logging.
        circuit_breakers (Dict[str, CircuitBreaker]): Circuit breakers by name.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the error handler.
        
        Args:
            logger: Logger instance to use.
        """
        self.logger = logger
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentError:
        """
        Handle an error with classification and logging.
        
        Args:
            error (Exception): The error to handle.
            context (Optional[Dict[str, Any]]): Additional context for logging.
            
        Returns:
            AgentError: The classified and wrapped error.
        """
        if isinstance(error, AgentError):
            agent_error = error
        else:
            agent_error = wrap_error(error)
        
        if self.logger:
            log_data = agent_error.to_dict()
            if context:
                log_data["context"] = context
            
            if agent_error.category == ErrorCategory.PERMANENT:
                self.logger.error(f"Permanent error: {log_data}")
            elif agent_error.is_retryable:
                self.logger.warning(f"Retryable error: {log_data}")
            else:
                self.logger.error(f"Error: {log_data}")
        
        return agent_error
    
    def get_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker by name.
        
        Args:
            name (str): Name of the circuit breaker.
            failure_threshold (int): Failures before opening.
            recovery_timeout (float): Seconds before recovery attempt.
            
        Returns:
            CircuitBreaker: The circuit breaker instance.
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
        return self.circuit_breakers[name]
    
    def get_all_circuit_breaker_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the state of all circuit breakers.
        
        Returns:
            Dict[str, Dict[str, Any]]: Circuit breaker states by name.
        """
        return {
            name: cb.to_dict()
            for name, cb in self.circuit_breakers.items()
        }


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler(logger=None) -> ErrorHandler:
    """
    Get the global error handler instance.
    
    Args:
        logger: Logger instance to use.
        
    Returns:
        ErrorHandler: The global error handler.
    """
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(logger=logger)
    return _error_handler