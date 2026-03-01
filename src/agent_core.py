"""
agent_core.py

This module contains the central orchestrator for the Agentic Workflow application.
It defines the `CustomerServiceAgent` class, which implements the ReAct (Reasoning and Acting)
pattern. The agent intelligently decides when to query the ClassicModels SQL database via
tools, and specifically implements a Reflection pattern to catch and auto-correct its own
SQL syntax errors without crashing.

Enhanced with comprehensive observability, tracing, metrics collection, and error handling
for production-grade monitoring and debugging capabilities.
"""

import time
import uuid
from typing import List, Dict, Any, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from src.config import APP_CONFIG, get_groq_api_key
from src.logger import setup_logger, sanitize_for_logging, get_logger
from src.tools_database import tools as SQL_TOOLS

# Import observability modules
from src.tracing import (
    start_trace, end_trace, start_span, complete_span, fail_span,
    get_trace_id, get_current_trace
)
from src.metrics import (
    get_metrics_collector, estimate_tokens
)
from src.observability import (
    ObservabilityContext, get_observability_manager
)
from src.error_handling import (
    AgentError, LLMError, ToolExecutionError, 
    classify_error, wrap_error, retry_on_error, RetryConfig, LLM_RETRY_CONFIG
)
from src.debug_utils import (
    DebugMode, ConversationState, MessageHistoryInspector,
    export_conversation_for_debugging
)

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
    4. Comprehensive observability with tracing, metrics, and structured logging.
    
    Attributes:
        api_key (str): The Groq API key for LLM access.
        model_name (str): The specific LLM model version being used (e.g., llama3-8b-8192).
        max_iterations (int): The hard limit on how many tool calls the agent can make per query.
        enable_reflection (bool): Toggles whether the agent is allowed to see and fix its own errors.
        llm (ChatGroq): The initialized LLM client block.
        llm_with_tools: The LLM client specifically bound to the SQL execution tools.
        tool_map (dict): A helper dictionary mapping tool names to actual python functions.
        debug_mode (DebugMode): Debug mode configuration.
        enable_observability (bool): Whether to enable observability features.
    """
    
    def __init__(
        self,
        debug_mode: bool = False,
        verbose: bool = False,
        enable_observability: bool = True
    ):
        """
        Initialize the Customer Service Agent.
        
        Args:
            debug_mode (bool): Enable debug mode for detailed output.
            verbose (bool): Enable verbose output in debug mode.
            enable_observability (bool): Enable observability features.
        """
        self.api_key = get_groq_api_key()
        self.model_name = APP_CONFIG.get("llm", {}).get("model", "llama-3.3-70b-versatile")
        self.max_iterations = APP_CONFIG.get("agent", {}).get("max_iterations", 5)
        self.enable_reflection = APP_CONFIG.get("agent", {}).get("enable_reflection", True)
        self.enable_observability = enable_observability
        
        # Initialize debug mode
        self.debug_mode = DebugMode(
            enabled=debug_mode,
            verbose=verbose,
            capture_messages=True,
            capture_tool_outputs=True
        )
        
        logger.info(f"Initializing Customer Service Agent with model '{self.model_name}'",
                   extra={"structured_data": {
                       "model": self.model_name,
                       "max_iterations": self.max_iterations,
                       "enable_reflection": self.enable_reflection,
                       "debug_mode": debug_mode
                   }})
        
        # Initialize Groq LLM and bind our tools
        self.llm = ChatGroq(
            api_key=self.api_key, 
            model=self.model_name,
            temperature=APP_CONFIG.get("llm", {}).get("temperature", 0.0)
        )
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        
        # Create a dictionary to easily call tools by name
        self.tool_map = {tool.name: tool for tool in ALL_TOOLS}
        
        # Get observability manager
        if self.enable_observability:
            self.observability_manager = get_observability_manager(
                logger=logger,
                debug_mode=debug_mode
            )

    def run(self, user_query: str) -> str:
        """
        Executes the main Reasoning-Acting-Observing (ReAct) loop for a given user query.
        
        The loop continues until either the LLM decides it has enough information to 
        answer the user without calling more tools, or the `max_iterations` limit is reached.
        If a tool fails (e.g., bad SQL syntax), the error is appended as a ToolMessage observation
        so the LLM can try again.
        
        Enhanced with comprehensive observability tracking including:
        - Trace context with correlation IDs
        - LLM call metrics (tokens, latency, cost)
        - Tool execution metrics
        - Debug state capture
        
        Args:
            user_query (str): The natural language question from the user.
            
        Returns:
            str: The final synthesized natural language answer from the LLM, or an error message.
        """
        # Generate unique query ID
        query_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize observability context
        obs_context: Optional[ObservabilityContext] = None
        conversation_state: Optional[ConversationState] = None
        
        if self.enable_observability:
            obs_context = self.observability_manager.create_context(
                user_query=user_query,
                metadata={"query_id": query_id}
            )
        
        # Initialize debug conversation state
        if self.debug_mode.should_capture():
            conversation_state = ConversationState(query_id)
            conversation_state.add_message("user", user_query)
        
        # Build system prompt
        system_prompt = (
            "You are a helpful customer service AI representing a Classic Vehicle Models company. "
            "You have access to a SQL database containing inventory, customers, and orders. "
            "CRITICAL INSTRUCTIONS: "
            "1. ALWAYS call get_database_schema() FIRST to understand the table structures before writing any SQL. "
            "2. After reviewing the schema, write a raw SQLite SELECT query using execute_sql_query(query). "
            "3. Never guess column names. Always verify with the schema. "
            "4. Do not return the raw JSON to the user. Synthesize the JSON data into a helpful, conversational answer."
        )
        
        messages = [SystemMessage(content=system_prompt)]
        messages.append(HumanMessage(content=user_query))
        
        logger.info(
            f"Starting agent execution for query: {sanitize_for_logging(user_query)}",
            extra={
                "structured_data": {
                    "query_id": query_id,
                    "query_preview": user_query[:100]
                },
                "trace_id": query_id
            }
        )

        final_response = None
        success = True
        
        try:
            for iteration in range(self.max_iterations):
                iteration_num = iteration + 1
                
                # Track iteration in observability
                if obs_context:
                    obs_context.start_iteration()
                
                # Track iteration in debug state
                if conversation_state:
                    conversation_state.increment_iteration()
                
                logger.info(
                    f"Iteration {iteration_num}",
                    extra={
                        "structured_data": {"iteration": iteration_num},
                        "trace_id": query_id
                    }
                )
                
                # 1. Reason / Plan - Invoke LLM
                llm_start_time = time.time()
                try:
                    response = self.llm_with_tools.invoke(messages)
                    llm_latency_ms = (time.time() - llm_start_time) * 1000
                    
                    # Record LLM call metrics
                    if obs_context:
                        # Estimate tokens from messages and response
                        input_text = str([m.content for m in messages])
                        output_text = response.content or ""
                        obs_context.record_llm_call(
                            model=self.model_name,
                            input_text=input_text,
                            output_text=output_text,
                            latency_ms=llm_latency_ms,
                            success=True
                        )
                    
                    # Track in debug state
                    if conversation_state:
                        conversation_state.add_message(
                            "assistant",
                            response.content or "",
                            iteration=iteration_num,
                            tool_calls=response.tool_calls
                        )
                    
                    logger.info(
                        f"LLM response received",
                        extra={
                            "structured_data": {
                                "latency_ms": round(llm_latency_ms, 2),
                                "has_tool_calls": bool(response.tool_calls),
                                "tool_call_count": len(response.tool_calls) if response.tool_calls else 0
                            },
                            "trace_id": query_id
                        }
                    )
                    
                except Exception as e:
                    llm_latency_ms = (time.time() - llm_start_time) * 1000
                    error = wrap_error(e, LLMError, f"LLM invocation failed: {str(e)}")
                    
                    # Record failed LLM call
                    if obs_context:
                        obs_context.record_llm_call(
                            model=self.model_name,
                            input_text="",
                            output_text="",
                            latency_ms=llm_latency_ms,
                            success=False,
                            error_message=str(e)
                        )
                    
                    logger.error(
                        f"LLM invocation failed: {str(e)}",
                        extra={
                            "structured_data": {
                                "error_type": type(e).__name__,
                                "latency_ms": round(llm_latency_ms, 2)
                            },
                            "trace_id": query_id
                        }
                    )
                    
                    if error.is_retryable:
                        # Could implement retry logic here
                        pass
                    
                    raise error
                
                messages.append(response)
                
                # If the LLM did not decide to call any tools, it's done processing.
                if not response.tool_calls:
                    logger.info(
                        "Agent concluded task (No tool calls returned)",
                        extra={"trace_id": query_id}
                    )
                    final_response = sanitize_for_logging(response.content)
                    break

                # 2. Act & Observe (Tool Execution)
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    logger.info(
                        f"LLM called tool '{tool_name}'",
                        extra={
                            "structured_data": {
                                "tool_name": tool_name,
                                "tool_args": str(tool_args)[:200]
                            },
                            "trace_id": query_id
                        }
                    )
                    
                    if tool_name not in self.tool_map:
                        error_msg = f"Error: Tool {tool_name} does not exist."
                        logger.warning(
                            error_msg,
                            extra={"trace_id": query_id}
                        )
                        messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                        
                        # Record failed tool call
                        if obs_context:
                            obs_context.record_tool_call(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                result=error_msg,
                                latency_ms=0,
                                success=False,
                                error_message=f"Tool {tool_name} not found"
                            )
                        
                        if conversation_state:
                            conversation_state.add_tool_call(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                result=error_msg,
                                success=False,
                                error=f"Tool not found"
                            )
                        continue
                    
                    # Execute the actual python tool
                    tool_start_time = time.time()
                    try:
                        tool_function = self.tool_map[tool_name]
                        tool_output = tool_function.invoke(tool_args)
                        tool_latency_ms = (time.time() - tool_start_time) * 1000
                        
                        tool_output_str = str(tool_output)
                        logger.info(
                            f"Tool '{tool_name}' executed successfully",
                            extra={
                                "structured_data": {
                                    "latency_ms": round(tool_latency_ms, 2),
                                    "result_size": len(tool_output_str)
                                },
                                "trace_id": query_id
                            }
                        )
                        
                        messages.append(ToolMessage(content=tool_output_str, tool_call_id=tool_id))
                        
                        # Record successful tool call
                        if obs_context:
                            obs_context.record_tool_call(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                result=tool_output_str,
                                latency_ms=tool_latency_ms,
                                success=True
                            )
                        
                        if conversation_state:
                            conversation_state.add_tool_call(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                result=tool_output_str,
                                success=True
                            )
                        
                    except Exception as e:
                        tool_latency_ms = (time.time() - tool_start_time) * 1000
                        error_msg = f"Tool Execution Error: {str(e)}"
                        
                        logger.error(
                            error_msg,
                            extra={
                                "structured_data": {
                                    "tool_name": tool_name,
                                    "latency_ms": round(tool_latency_ms, 2),
                                    "error_type": type(e).__name__
                                },
                                "trace_id": query_id
                            }
                        )
                        
                        # Record failed tool call
                        if obs_context:
                            obs_context.record_tool_call(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                result=error_msg,
                                latency_ms=tool_latency_ms,
                                success=False,
                                error_message=str(e)
                            )
                        
                        if conversation_state:
                            conversation_state.add_tool_call(
                                tool_name=tool_name,
                                tool_args=tool_args,
                                result=error_msg,
                                success=False,
                                error=str(e)
                            )
                        
                        # 3. Reflection (M2 Pattern)
                        if self.enable_reflection:
                            logger.info(
                                "Reflection triggered. Sending error back to LLM to auto-correct.",
                                extra={"trace_id": query_id}
                            )
                            messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                        else:
                            success = False
                            final_response = f"Failed execution: {error_msg}"
                            break
                
                # Check if we need to break out of the iteration loop
                if not success:
                    break
            
            else:
                # Loop completed without break (max iterations reached)
                logger.warning(
                    f"Agent reached max iterations ({self.max_iterations}) and was forcibly stopped.",
                    extra={"trace_id": query_id}
                )
                final_response = "Agent stopped due to reaching iteration limits."
                success = False
                
        except Exception as e:
            success = False
            final_response = f"Agent execution failed: {str(e)}"
            logger.error(
                f"Agent execution failed with error: {str(e)}",
                extra={
                    "structured_data": {
                        "error_type": type(e).__name__
                    },
                    "trace_id": query_id
                },
                exc_info=True
            )
        
        finally:
            # Calculate total latency
            total_latency_ms = (time.time() - start_time) * 1000
            
            # Finalize observability context
            if obs_context:
                obs_context.finalize(success=success, final_response=final_response)
                self.observability_manager.remove_context(query_id)
            
            # Finalize and export debug state if enabled
            if conversation_state:
                conversation_state.finalize(final_response or "")
                if self.debug_mode.should_capture():
                    export_files = export_conversation_for_debugging(
                        state=conversation_state,
                        output_dir=self.debug_mode.output_dir
                    )
                    logger.info(
                        f"Debug export completed",
                        extra={
                            "structured_data": {"export_files": export_files},
                            "trace_id": query_id
                        }
                    )
            
            # Log final summary
            logger.info(
                f"Agent execution completed",
                extra={
                    "structured_data": {
                        "success": success,
                        "total_latency_ms": round(total_latency_ms, 2),
                        "response_preview": final_response[:100] if final_response else None
                    },
                    "trace_id": query_id
                }
            )
        
        return final_response or "No response generated."

    def run_with_metrics(self, user_query: str) -> Dict[str, Any]:
        """
        Execute a query and return both the response and detailed metrics.
        
        This method is useful for evaluation and monitoring purposes,
        providing visibility into the agent's execution details.
        
        Args:
            user_query (str): The natural language question from the user.
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - response: The final response text
                - metrics: Execution metrics including timing, tokens, and costs
                - trace_id: The correlation ID for this execution
        """
        query_id = str(uuid.uuid4())
        
        # Run the agent
        response = self.run(user_query)
        
        # Collect metrics
        metrics_summary = {}
        if self.enable_observability:
            collector = get_metrics_collector()
            # Find the query metrics by query_id
            for qm in collector.queries:
                if qm.query_id == query_id:
                    metrics_summary = qm.to_dict()
                    break
        
        return {
            "response": response,
            "metrics": metrics_summary,
            "trace_id": query_id
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.
        
        Returns:
            Dict[str, Any]: Summary of metrics across all queries.
        """
        if not self.enable_observability:
            return {"error": "Observability is disabled"}
        
        collector = get_metrics_collector()
        return collector.get_summary()

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get metrics for recent queries.
        
        Args:
            limit (int): Maximum number of queries to return.
            
        Returns:
            List[Dict[str, Any]]: List of recent query metrics.
        """
        if not self.enable_observability:
            return []
        
        collector = get_metrics_collector()
        return collector.get_recent_queries(limit)