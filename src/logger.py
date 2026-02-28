"""
logger.py

This module provides an application-wide, centralized logging system. It ensures that 
all runtime events, errors, and LLM thoughts are recorded gracefully to both the 
console and a persistent log file (`logs/agent_run.log`). It also implements ASCII
sanitization to prevent encoding crashes when dealing with unpredictable LLM output.
"""

import os
import sys
import logging
from pathlib import Path

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
    level_str = config.get("logging", {}).get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. File Handler
    log_file_path = config.get("logging", {}).get("file_path", "logs/agent_run.log")
    log_dir = Path(log_file_path).parent
    # Ensure logs directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file_path, encoding='ascii', errors='replace')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Console Handler (Optional)
    if config.get("logging", {}).get("console_output", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def sanitize_for_logging(text: str) -> str:
    """
    Sanitizes string outputs to strictly ASCII to prevent terminal rendering errors.
    Useful for LLM outputs.
    """
    if not isinstance(text, str):
        return str(text)
    return text.encode('ascii', 'replace').decode('ascii')
