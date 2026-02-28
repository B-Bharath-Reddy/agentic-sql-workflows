"""
main.py

This module serves as the primary Command Line Interface (CLI) entry point for the Agentic Workflow application.
It is responsible for initializing the environment, setting up the root logger, instantiating the 
CustomerServiceAgent core, and managing the interactive user input/output loop.

Usage:
    python main.py
"""

import sys
from dotenv import load_dotenv
from src.agent_core import CustomerServiceAgent
from src.logger import setup_logger
from src.config import APP_CONFIG

def main():
    """
    The main execution function for the CLI agent.
    
    Workflow:
    1. Loads environment variables (e.g., GROQ_API_KEY) from a local .env file.
    2. Initializes the centralized logging system based on the application configuration.
    3. Attempts to initialize the CustomerServiceAgent. Exits gracefully if API keys are missing.
    4. Enters an infinite while loop to accept user queries via stdin.
    5. Passes queries to the agent, captures the reasoning/acting loop, and outputs the final response.
    6. Handles KeyboardInterrupts (Ctrl+C) to allow for safe shutdown.
    """
    # Attempt to load env vars immediately
    load_dotenv()
    
    logger = setup_logger(APP_CONFIG)
    logger.info("Starting up the CLI Agent Workspace.")
    
    try:
        agent = CustomerServiceAgent()
    except Exception as e:
        print(f"Failed to initialize the agent. Error: {e}")
        print("Please make sure you have created a .env file with your GROQ_API_KEY")
        sys.exit(1)

    print("\n" + "="*50)
    print("Welcome to the Agentic Workflow CLI")
    print("Type your query (or type 'quit' to exit)")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("User >> ")
            if user_input.strip().lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input.strip():
                continue
                
            print("\nThinking...")
            final_answer = agent.run(user_input)
            print(f"\nAgent >> {final_answer}\n")
            
        except KeyboardInterrupt:
            # Handle Ctrl+C smoothly
            break
        except Exception as e:
            logger.error(f"Critical Application Error: {e}")
            print(f"\nAn error occurred: {e}\n")

    print("\nAgent shut down successfully.")

if __name__ == "__main__":
    main()
