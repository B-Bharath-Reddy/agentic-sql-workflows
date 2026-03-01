"""
logger.py

This module provides an application-wide, centralized logging system. It ensures that 
all runtime events, errors, and LLM thoughts are recorded gracefully to both the 
console and a persistent log file (`logs/agent_run.log`). It also implements ASCII
sanitization to prevent encoding crashes when dealing with unpredictable LLM output.

Enhanced with structured logging capabilities, JSON output support, and integration
with the tracing module for correlation ID tracking.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that supports both standard text and structured JSON output.
    
    This formatter can include trace context (correlation IDs) in log messages
    and output logs in JSON format for easier parsing by log aggregation systems.
    
    Attributes:
        include_trace (bool): Whether to include trace context in logs.
        json_format (bool): Whether to output logs in JSON format.
    """
    
    def __init__(
        self,
        include_trace: bool = True,
        json_format: bool = False,
        *args,
        **kwargs
    ):
        """
        Initialize the structured formatter.
        
        Args:
            include_trace (bool): Include trace context in logs.
            json_format (bool): Output logs in JSON format.
            *args: Additional positional arguments for parent class.
            **kwargs: Additional keyword arguments for parent class.
        """
        super().__init__(*args, **kwargs)
        self.include_trace = include_trace
        self.json_format = json_format
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record.
        
        Args:
            record (logging.LogRecord): The log record to format.
            
        Returns:
            str: The formatted log message.
        """
        # Add trace context if available
        trace_id = getattr(record, 'trace_id', None)
        span_id = getattr(record, 'span_id', None)
        
        if self.json_format:
            return self._format_json(record, trace_id, span_id)
        else:
            return self._format_text(record, trace_id, span_id)
    
    def _format_text(
        self,
        record: logging.LogRecord,
        trace_id: Optional[str],
        span_id: Optional[str]
    ) -> str:
        """
        Format the log record as text.
        
        Args:
            record (logging.LogRecord): The log record to format.
            trace_id (Optional[str]): The trace ID if available.
            span_id (Optional[str]): The span ID if available.
            
        Returns:
            str: The formatted log message.
        """
        timestamp = self.formatTime(record, self.datefmt)
        
        # Build trace context string
        trace_context = ""
        if self.include_trace and trace_id:
            trace_context = f"[trace:{trace_id[:8]}] "
            if span_id:
                trace_context = f"[trace:{trace_id[:8]}|span:{span_id[:8]}] "
        
        # Get structured data if present
        structured_data = getattr(record, 'structured_data', None)
        
        # Format base message
        message = f"[{timestamp}] {record.levelname:8} - {record.name} - {trace_context}{record.getMessage()}"
        
        # Add structured data if present
        if structured_data:
            try:
                data_str = json.dumps(structured_data, default=str, indent=None)
                message = f"{message} | {data_str}"
            except (TypeError, ValueError):
                pass
        
        # Add exception info if present
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        
        return message
    
    def _format_json(
        self,
        record: logging.LogRecord,
        trace_id: Optional[str],
        span_id: Optional[str]
    ) -> str:
        """
        Format the log record as JSON.
        
        Args:
            record (logging.LogRecord): The log record to format.
            trace_id (Optional[str]): The trace ID if available.
            span_id (Optional[str]): The span ID if available.
            
        Returns:
            str: The formatted log message as JSON.
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add trace context
        if self.include_trace:
            if trace_id:
                log_data["trace_id"] = trace_id
            if span_id:
                log_data["span_id"] = span_id
        
        # Add structured data
        structured_data = getattr(record, 'structured_data', None)
        if structured_data:
            log_data["data"] = structured_data
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class StructuredLogger(logging.Logger):
    """
    Enhanced logger class that supports structured logging with trace context.
    
    This class extends the standard logger to support structured data in log
    messages and automatic trace context injection.
    """
    
    def _log_structured(
        self,
        level: int,
        msg: str,
        structured_data: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        """
        Log a message with structured data and trace context.
        
        Args:
            level (int): The log level.
            msg (str): The log message.
            structured_data (Optional[Dict[str, Any]]): Structured data to include.
            trace_id (Optional[str]): The trace ID for correlation.
            span_id (Optional[str]): The span ID for correlation.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Try to get trace context from the current context if not provided
        if trace_id is None:
            try:
                from src.tracing import get_trace_id
                trace_id = get_trace_id()
            except ImportError:
                pass
        
        # Create extra dict with trace context and structured data
        extra = kwargs.pop('extra', {})
        if trace_id:
            extra['trace_id'] = trace_id
        if span_id:
            extra['span_id'] = span_id
        if structured_data:
            extra['structured_data'] = structured_data
        
        kwargs['extra'] = extra
        self.log(level, msg, *args, **kwargs)
    
    def debug_structured(
        self,
        msg: str,
        structured_data: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log a debug message with structured data.
        
        Args:
            msg (str): The log message.
            structured_data (Optional[Dict[str, Any]]): Structured data to include.
            trace_id (Optional[str]): The trace ID for correlation.
            span_id (Optional[str]): The span ID for correlation.
            **kwargs: Additional keyword arguments.
        """
        self._log_structured(logging.DEBUG, msg, structured_data, trace_id, span_id, **kwargs)
    
    def info_structured(
        self,
        msg: str,
        structured_data: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log an info message with structured data.
        
        Args:
            msg (str): The log message.
            structured_data (Optional[Dict[str, Any]]): Structured data to include.
            trace_id (Optional[str]): The trace ID for correlation.
            span_id (Optional[str]): The span ID for correlation.
            **kwargs: Additional keyword arguments.
        """
        self._log_structured(logging.INFO, msg, structured_data, trace_id, span_id, **kwargs)
    
    def warning_structured(
        self,
        msg: str,
        structured_data: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log a warning message with structured data.
        
        Args:
            msg (str): The log message.
            structured_data (Optional[Dict[str, Any]]): Structured data to include.
            trace_id (Optional[str]): The trace ID for correlation.
            span_id (Optional[str]): The span ID for correlation.
            **kwargs: Additional keyword arguments.
        """
        self._log_structured(logging.WARNING, msg, structured_data, trace_id, span_id, **kwargs)
    
    def error_structured(
        self,
        msg: str,
        structured_data: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log an error message with structured data.
        
        Args:
            msg (str): The log message.
            structured_data (Optional[Dict[str, Any]]): Structured data to include.
            trace_id (Optional[str]): The trace ID for correlation.
            span_id (Optional[str]): The span ID for correlation.
            **kwargs: Additional keyword arguments.
        """
        self._log_structured(logging.ERROR, msg, structured_data, trace_id, span_id, **kwargs)
    
    def critical_structured(
        self,
        msg: str,
        structured_data: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log a critical message with structured data.
        
        Args:
            msg (str): The log message.
            structured_data (Optional[Dict[str, Any]]): Structured data to include.
            trace_id (Optional[str]): The trace ID for correlation.
            span_id (Optional[str]): The span ID for correlation.
            **kwargs: Additional keyword arguments.
        """
        self._log_structured(logging.CRITICAL, msg, structured_data, trace_id, span_id, **kwargs)


# Set the custom logger class
logging.setLoggerClass(StructuredLogger)


def setup_logger(config: dict) -> logging.Logger:
    """
    Sets up a centralized logger that writes to both the console and a log file.
    Ensures that output is clean and ASCII-compliant.

    Args:
        config (dict): Configuration dictionary loaded from config.yaml.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("AgenticWorkflow")
    
    # Avoid duplicate logs if instantiated multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set base level
    logging_config = config.get("logging", {})
    level_str = logging_config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    # Determine if JSON format should be used
    json_format = logging_config.get("json_format", False)
    include_trace = logging_config.get("include_trace", True)
    
    # Create the appropriate formatter
    if json_format:
        formatter = StructuredFormatter(
            include_trace=include_trace,
            json_format=True
        )
    else:
        formatter = StructuredFormatter(
            include_trace=include_trace,
            json_format=False,
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    # 1. File Handler
    log_file_path = logging_config.get("file_path", "logs/agent_run.log")
    log_dir = Path(log_file_path).parent
    # Ensure logs directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # 2. Console Handler (Optional)
    if logging_config.get("console_output", True):
        console_handler = logging.StreamHandler(sys.stdout)
        # Use a simpler formatter for console (no JSON)
        console_formatter = StructuredFormatter(
            include_trace=include_trace,
            json_format=False,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # 3. Debug File Handler (Optional - for verbose debugging)
    if logging_config.get("debug_file", False):
        debug_file_path = logging_config.get("debug_file_path", "logs/debug.log")
        debug_dir = Path(debug_file_path).parent
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_handler = logging.FileHandler(debug_file_path, encoding='utf-8')
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

    return logger


def sanitize_for_logging(text: str) -> str:
    """
    Sanitizes string outputs to strictly ASCII to prevent terminal rendering errors.
    Useful for LLM outputs.
    
    Args:
        text (str): The text to sanitize.
        
    Returns:
        str: The sanitized text.
    """
    if not isinstance(text, str):
        return str(text)
    return text.encode('ascii', 'replace').decode('ascii')


def get_logger(name: str = "AgenticWorkflow") -> StructuredLogger:
    """
    Get a logger instance by name.
    
    This function returns a StructuredLogger instance that supports
    structured logging with trace context.
    
    Args:
        name (str): The name of the logger to get.
        
    Returns:
        StructuredLogger: The logger instance.
    """
    return logging.getLogger(name)


def log_with_context(
    logger: logging.Logger,
    level: int,
    msg: str,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    **kwargs
) -> None:
    """
    Convenience function to log with trace context.
    
    Args:
        logger (logging.Logger): The logger to use.
        level (int): The log level.
        msg (str): The log message.
        trace_id (Optional[str]): The trace ID for correlation.
        span_id (Optional[str]): The span ID for correlation.
        **kwargs: Additional keyword arguments.
    """
    extra = kwargs.pop('extra', {})
    if trace_id:
        extra['trace_id'] = trace_id
    if span_id:
        extra['span_id'] = span_id
    kwargs['extra'] = extra
    logger.log(level, msg, **kwargs)