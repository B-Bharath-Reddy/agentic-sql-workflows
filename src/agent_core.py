"""
agent_core.py

This module contains the central orchestrator for the Agentic Workflow application.
It defines the `CustomerServiceAgent` class, which implements the ReAct (Reasoning and Acting)
pattern. The agent intelligently decides when to query the ClassicModels SQL database via
tools, and specifically implements a Reflection pattern to catch and auto-correct its own
SQL syntax errors without crashing.
"""

import json
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from src.config import APP_CONFIG, get_groq_api_key
from src.logger import setup_logger, sanitize_for_logging
from src.tools_database import tools as SQL_TOOLS

logger = setup_logger(APP_CONFIG)

ALL_TOOLS = SQL_TOOLS

class CustomerServiceAgent:
    """
    An autonomous agent that processes natural language queries about the ClassicModels database.
    
    This class is responsible for:
    1. Managing the LangChain conversation history.
    2. Executing a ReAct loop (Think -> Act -> Observe).
    3. Catching tool execution errors (like sqlite3.OperationalError) and feeding them 
       back to the LLM to trigger self-reflection and auto-correction.
    
    Attributes:
        api_key (str): The Groq API key for LLM access.
        model_name (str): The specific LLM model version being used (e.g., llama3-8b-8192).
        max_iterations (int): The hard limit on how many tool calls the agent can make per query.
        enable_reflection (bool): Toggles whether the agent is allowed to see and fix its own errors.
        llm (ChatGroq): The initialized LLM client block.
        llm_with_tools: The LLM client specifically bound to the SQL execution tools.
        tool_map (dict): A helper dictionary mapping tool names to actual python functions.
    """
    def __init__(self):
        self.api_key = get_groq_api_key()
        self.model_name = APP_CONFIG.get("llm", {}).get("model", "llama-3.3-70b-versatile")
        self.max_iterations = APP_CONFIG.get("agent", {}).get("max_iterations", 5)
        self.enable_reflection = APP_CONFIG.get("agent", {}).get("enable_reflection", True)
        
        logger.info(f"Initializing Customer Service Agent with model '{self.model_name}'")
        
        # Initialize Groq LLM and bind our tools
        self.llm = ChatGroq(
            api_key=self.api_key, 
            model=self.model_name,
            temperature=APP_CONFIG.get("llm", {}).get("temperature", 0.0)
        )
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        
        # Create a dictionary to easily call tools by name
        self.tool_map = {tool.name: tool for tool in ALL_TOOLS}

    def run(self, user_query: str) -> str:
        """
        Executes the main Reasoning-Acting-Observing (ReAct) loop for a given user query.
        
        The loop continues until either the LLM decides it has enough information to 
        answer the user without calling more tools, or the `max_iterations` limit is reached.
        If a tool fails (e.g., bad SQL syntax), the error is appended as a ToolMessage observation
        so the LLM can try again.
        
        Args:
            user_query (str): The natural language question from the user.
            
        Returns:
            str: The final synthesized natural language answer from the LLM, or an error message.
        """
        messages = [
            SystemMessage(content="You are a helpful customer service AI representing a Classic Vehicle Models company. "
                                  "You have access to a SQL database containing inventory, customers, and orders. "
                                  "CRITICAL INSTRUCTIONS: "
                                  "1. ALWAYS call get_database_schema() FIRST to understand the table structures before writing any SQL. "
                                  "2. After reviewing the schema, write a raw SQLite SELECT query using execute_sql_query(query). "
                                  "3. Never guess column names. Always verify with the schema. "
                                  "4. Do not return the raw JSON to the user. Synthesize the JSON data into a helpful, conversational answer.")
        ]
        
        messages.append(HumanMessage(content=user_query))
        logger.info(f"Starting agent execution for query: {sanitize_for_logging(user_query)}")

        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}...")
            
            # 1. Reason / Plan
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # If the LLM did not decide to call any tools, it's done processing.
            if not response.tool_calls:
                logger.info("Agent concluded task (No tool calls returned).")
                return sanitize_for_logging(response.content)

            # 2. Act & Observe (Tool Execution)
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                logger.info(f"LLM called tool '{tool_name}' with args {tool_args}")
                
                if tool_name not in self.tool_map:
                    error_msg = f"Error: Tool {tool_name} does not exist."
                    logger.warning(error_msg)
                    messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                    continue
                
                # Execute the actual python tool
                try:
                    tool_function = self.tool_map[tool_name]
                    tool_output = tool_function.invoke(tool_args)
                    logger.info(f"Tool '{tool_name}' returned: {sanitize_for_logging(str(tool_output))}")
                    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))
                    
                except Exception as e:
                    # 3. Reflection (M2 Pattern)
                    error_msg = f"Tool Execution Error: {str(e)}"
                    logger.error(error_msg)
                    if self.enable_reflection:
                        logger.info("Reflection triggered. Sending error back to LLM to auto-correct.")
                        messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                    else:
                        return f"Failed execution: {error_msg}"

        logger.warning(f"Agent reached max iterations ({self.max_iterations}) and was forcibly stopped.")
        return "Agent stopped due to reaching iteration limits."
