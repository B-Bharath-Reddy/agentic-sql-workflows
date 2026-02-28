"""
config.py

This module is responsible for loading and validating the application's configuration
settings. It reads from a central `config.yaml` file to configure model selection,
agent properties, and logging settings. It also safely retrieves necessary API keys
from the system environment variables.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

def load_config() -> dict:
    """
    Loads configuration settings from the config.yaml file.
    Returns:
        dict: A dictionary containing the application settings.
    """
    root_dir = Path(__file__).parent.parent
    config_path = root_dir / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configuration file at {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def get_groq_api_key() -> str:
    """
    Safely retrieves the Groq API key from environment variables.
    Returns:
        str: The API key.
    Raises:
        ValueError: If the key is not found in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running the agent.")
    return api_key

# Singleton configuration dict for easy import
APP_CONFIG = load_config()
