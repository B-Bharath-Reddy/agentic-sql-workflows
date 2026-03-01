"""
tools_database.py

This module contains the dynamic SQL Langchain Tools used by the agent to interact
with the local ClassicModels SQLite database. It provides functionalities to inspect
the database schema and deliberately execute raw SQL queries, enforcing safety checks
(read-only) and size limits.

Enhanced with observability integration for tracing, metrics collection, and
comprehensive error handling.
"""

import sqlite3
import json
import time
import os
from typing import Optional, Dict, Any, List

from langchain_core.tools import tool

from src.config import APP_CONFIG
from src.logger import get_logger

# Import observability modules
from src.tracing import start_span, complete_span, fail_span, get_current_trace
from src.error_handling import (
    DatabaseError, ToolExecutionError, 
    classify_error, wrap_error
)

logger = get_logger("AgenticWorkflow")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "classicmodels.db")


def _get_db_connection() -> sqlite3.Connection:
    """
    Create and return a database connection.
    
    Returns:
        sqlite3.Connection: The database connection.
        
    Raises:
        DatabaseError: If connection fails.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except sqlite3.Error as e:
        raise DatabaseError(
            message=f"Failed to connect to database: {str(e)}",
            category=classify_error(e)
        )


def _execute_with_observability(
    operation_name: str,
    operation_func,
    *args,
    **kwargs
) -> str:
    """
    Execute a database operation with observability tracking.
    
    This wrapper adds tracing spans and timing to database operations.
    
    Args:
        operation_name (str): Name of the operation for tracing.
        operation_func: The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
        
    Returns:
        str: The result of the operation.
    """
    start_time = time.time()
    span = None
    
    # Create tracing span if we're in a trace context
    trace = get_current_trace()
    if trace:
        span = start_span(f"db_{operation_name}", {
            "operation": operation_name,
            "db_path": DB_PATH
        })
    
    try:
        result = operation_func(*args, **kwargs)
        
        # Complete the span
        if span:
            latency_ms = (time.time() - start_time) * 1000
            complete_span(span, {
                "latency_ms": round(latency_ms, 2),
                "success": True
            })
        
        logger.debug(
            f"Database operation '{operation_name}' completed",
            extra={
                "structured_data": {
                    "operation": operation_name,
                    "latency_ms": round((time.time() - start_time) * 1000, 2)
                }
            }
        )
        
        return result
        
    except Exception as e:
        # Fail the span
        if span:
            latency_ms = (time.time() - start_time) * 1000
            fail_span(span, str(e))
        
        logger.error(
            f"Database operation '{operation_name}' failed: {str(e)}",
            extra={
                "structured_data": {
                    "operation": operation_name,
                    "error_type": type(e).__name__,
                    "latency_ms": round((time.time() - start_time) * 1000, 2)
                }
            }
        )
        
        raise


@tool
def get_database_schema() -> str:
    """
    Retrieves the schema information (tables and columns) for the entire database.
    
    This tool MUST be called FIRST by the agent to understand what tables exist and 
    what exact column names are available before attempting to generate any SQL.
    
    Returns:
        str: A formatted string listing all tables and their respective columns.
             Returns an error string if the database cannot be accessed.
    """
    def _get_schema_internal() -> str:
        conn = None
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = []
            table_count = 0
            total_columns = 0
            
            for table in tables:
                table_name = table[0]
                # Ignore sqlite internal tables
                if table_name.startswith("sqlite_"):
                    continue
                    
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                # Format: 'table_name' (column1, column2, ...)
                col_names = [col[1] for col in columns]
                schema_info.append(f"Table: '{table_name}' | Columns: {', '.join(col_names)}")
                
                table_count += 1
                total_columns += len(col_names)
            
            logger.info(
                "Schema retrieved successfully",
                extra={
                    "structured_data": {
                        "table_count": table_count,
                        "total_columns": total_columns
                    }
                }
            )
            
            return "\n".join(schema_info)
            
        except sqlite3.Error as e:
            error = wrap_error(e, DatabaseError, f"Error retrieving schema: {str(e)}")
            return f"Error retrieving schema: {str(e)}"
        except Exception as e:
            return f"Error retrieving schema: {str(e)}"
        finally:
            if conn:
                conn.close()
    
    return _execute_with_observability("get_schema", _get_schema_internal)


@tool
def execute_sql_query(query: str) -> str:
    """
    Executes a raw SQLite SELECT query against the ClassicModels database.
    
    The agent MUST use get_database_schema() prior to calling this to ensure accurate
    table and column naming. This function enforces strict read-only execution by
    blocking destructive keywords (DROP, DELETE, UPDATE, etc.) and truncates massive
    results to prevent LLM context-window overflow.
    
    Args:
        query (str): The raw SQLite SELECT statement to execute. 
                     Example: "SELECT productName FROM products LIMIT 5"
    
    Returns:
        str: A JSON formatted string containing the rows returned by the database,
             or a descriptive error message (e.g., sqlite3.OperationalError) if the 
             syntax is invalid, which the agent can use to reflect and auto-correct.
    """
    def _execute_query_internal() -> str:
        # Input validation
        if not query or not isinstance(query, str):
            return "Error: Query must be a non-empty string."
        
        # Log the query being executed
        logger.info(
            f"Executing SQL query",
            extra={
                "structured_data": {
                    "query_preview": query[:100] + "..." if len(query) > 100 else query
                }
            }
        )
        
        # Security check: Prevent destructive operations
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]
        upper_query = query.upper()
        
        for keyword in forbidden_keywords:
            if keyword in upper_query:
                logger.warning(
                    f"Blocked forbidden SQL keyword: {keyword}",
                    extra={
                        "structured_data": {
                            "keyword": keyword,
                            "query_preview": query[:50]
                        }
                    }
                )
                return f"Error: Only SELECT queries are permitted. Detected forbidden keyword: {keyword}"
        
        conn = None
        try:
            conn = _get_db_connection()
            # Return rows as dictionaries instead of tuples for nicer JSON output
            conn.row_factory = sqlite3.Row 
            cursor = conn.cursor()
            
            # Execute the query
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Format the output cleanly
            if not rows:
                logger.info(
                    "Query executed successfully, returned 0 rows",
                    extra={"structured_data": {"row_count": 0}}
                )
                return "Query executed successfully, but returned 0 rows."
            
            # Convert sqlite3.Row objects to dicts
            dict_rows = [dict(row) for row in rows]
            
            # Return as a JSON formatted string for the LLM to read easily
            result_str = json.dumps(dict_rows, indent=2)
            
            # Safety limit to prevent exploding the LLM context size
            max_result_size = APP_CONFIG.get("database", {}).get("max_result_size", 3000)
            
            if len(result_str) > max_result_size:
                truncated_rows = dict_rows[:5]
                truncated_result = json.dumps(truncated_rows, indent=2)
                truncated_result += f"\n... (Results truncated. Showing 5 of {len(dict_rows)} rows. Please refine your query with LIMIT or WHERE.)"
                
                logger.info(
                    "Query results truncated",
                    extra={
                        "structured_data": {
                            "total_rows": len(dict_rows),
                            "shown_rows": 5,
                            "result_size": len(result_str)
                        }
                    }
                )
                
                return truncated_result
            
            logger.info(
                "Query executed successfully",
                extra={
                    "structured_data": {
                        "row_count": len(dict_rows),
                        "result_size": len(result_str)
                    }
                }
            )
            
            return result_str
            
        except sqlite3.OperationalError as e:
            # M2 Pattern: Explicitly return the syntax error to the LLM so it can reflect and fix it
            error_msg = str(e)
            
            logger.warning(
                f"SQLite Operational Error: {error_msg}",
                extra={
                    "structured_data": {
                        "error_type": "OperationalError",
                        "query_preview": query[:100]
                    }
                }
            )
            
            return f"SQLite Syntax/Operational Error: {error_msg}\nPlease check your query syntax and table names (use get_database_schema if needed) and try again."
            
        except sqlite3.DatabaseError as e:
            error_msg = str(e)
            
            logger.error(
                f"Database Error: {error_msg}",
                extra={
                    "structured_data": {
                        "error_type": "DatabaseError",
                        "query_preview": query[:100]
                    }
                }
            )
            
            return f"Database Error: {error_msg}\nPlease verify your query and try again."
            
        except Exception as e:
            error_msg = str(e)
            
            logger.error(
                f"Unexpected execution error: {error_msg}",
                extra={
                    "structured_data": {
                        "error_type": type(e).__name__,
                        "query_preview": query[:100]
                    }
                },
                exc_info=True
            )
            
            return f"Execution Error: {error_msg}"
            
        finally:
            if conn:
                conn.close()
    
    return _execute_with_observability("execute_query", _execute_query_internal)


@tool
def get_table_sample(table_name: str, limit: int = 3) -> str:
    """
    Retrieves a sample of rows from a specific table to understand data patterns.
    
    This tool is useful for understanding the actual data in a table, including
    data types, value patterns, and common values. Use this after get_database_schema()
    to better understand the data before writing complex queries.
    
    Args:
        table_name (str): The name of the table to sample.
        limit (int): Maximum number of rows to return (default: 3, max: 10).
    
    Returns:
        str: A JSON formatted string containing sample rows from the table,
             or an error message if the table doesn't exist.
    """
    def _get_sample_internal() -> str:
        # Validate inputs
        if not table_name or not isinstance(table_name, str):
            return "Error: table_name must be a non-empty string."
        
        # Sanitize table name to prevent SQL injection
        if not table_name.replace("_", "").isalnum():
            return "Error: Invalid table name. Only alphanumeric characters and underscores are allowed."
        
        # Enforce limit bounds
        limit = max(1, min(limit, 10))
        
        logger.info(
            f"Getting table sample for '{table_name}'",
            extra={
                "structured_data": {
                    "table_name": table_name,
                    "limit": limit
                }
            }
        )
        
        conn = None
        try:
            conn = _get_db_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # First verify the table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (table_name,)
            )
            if not cursor.fetchone():
                return f"Error: Table '{table_name}' does not exist. Use get_database_schema() to see available tables."
            
            # Get sample rows
            query = f"SELECT * FROM {table_name} LIMIT {limit};"
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                return f"Table '{table_name}' exists but contains no rows."
            
            dict_rows = [dict(row) for row in rows]
            result_str = json.dumps(dict_rows, indent=2)
            
            logger.info(
                f"Table sample retrieved successfully",
                extra={
                    "structured_data": {
                        "table_name": table_name,
                        "row_count": len(dict_rows)
                    }
                }
            )
            
            return result_str
            
        except sqlite3.Error as e:
            logger.error(
                f"Database error getting table sample: {str(e)}",
                extra={
                    "structured_data": {
                        "table_name": table_name,
                        "error_type": type(e).__name__
                    }
                }
            )
            return f"Database Error: {str(e)}"
        except Exception as e:
            logger.error(
                f"Unexpected error getting table sample: {str(e)}",
                extra={
                    "structured_data": {
                        "table_name": table_name,
                        "error_type": type(e).__name__
                    }
                }
            )
            return f"Error: {str(e)}"
        finally:
            if conn:
                conn.close()
    
    return _execute_with_observability("get_table_sample", _get_sample_internal)


@tool
def get_table_row_count(table_name: str) -> str:
    """
    Gets the number of rows in a specific table.
    
    This tool is useful for understanding the size of tables before writing
    queries that might return large result sets.
    
    Args:
        table_name (str): The name of the table to count rows for.
    
    Returns:
        str: A message indicating the row count, or an error message.
    """
    def _get_count_internal() -> str:
        # Validate input
        if not table_name or not isinstance(table_name, str):
            return "Error: table_name must be a non-empty string."
        
        # Sanitize table name
        if not table_name.replace("_", "").isalnum():
            return "Error: Invalid table name. Only alphanumeric characters and underscores are allowed."
        
        conn = None
        try:
            conn = _get_db_connection()
            cursor = conn.cursor()
            
            # First verify the table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (table_name,)
            )
            if not cursor.fetchone():
                return f"Error: Table '{table_name}' does not exist. Use get_database_schema() to see available tables."
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            
            logger.info(
                f"Row count retrieved for '{table_name}'",
                extra={
                    "structured_data": {
                        "table_name": table_name,
                        "row_count": count
                    }
                }
            )
            
            return f"Table '{table_name}' contains {count:,} rows."
            
        except sqlite3.Error as e:
            logger.error(
                f"Database error getting row count: {str(e)}",
                extra={
                    "structured_data": {
                        "table_name": table_name,
                        "error_type": type(e).__name__
                    }
                }
            )
            return f"Database Error: {str(e)}"
        except Exception as e:
            logger.error(
                f"Unexpected error getting row count: {str(e)}",
                extra={
                    "structured_data": {
                        "table_name": table_name,
                        "error_type": type(e).__name__
                    }
                }
            )
            return f"Error: {str(e)}"
        finally:
            if conn:
                conn.close()
    
    return _execute_with_observability("get_table_row_count", _get_count_internal)


# Export the active tools
tools = [get_database_schema, execute_sql_query, get_table_sample, get_table_row_count]