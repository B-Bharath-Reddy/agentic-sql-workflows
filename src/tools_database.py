"""
tools_database.py

This module contains the dynamic SQL Langchain Tools used by the agent to interact
with the local ClassicModels SQLite database. It provides functionalities to inspect
the database schema and deliberately execute raw SQL queries, enforcing safety checks
(read-only) and size limits.
"""

import sqlite3
import json
from langchain_core.tools import tool
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "classicmodels.db")

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
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_info = []
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
            
        conn.close()
        return "\n".join(schema_info)
    except Exception as e:
        return f"Error retrieving schema: {str(e)}"

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
    # Security check: Prevent destructive operations
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
    upper_query = query.upper()
    if any(keyword in upper_query for keyword in forbidden_keywords):
         return "Error: Only SELECT queries are permitted."
         
    try:
        conn = sqlite3.connect(DB_PATH)
        # Return rows as dictionaries instead of tuples for nicer JSON output
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Format the output cleanly
        if not rows:
            return "Query executed successfully, but returned 0 rows."
            
        # Convert sqlite3.Row objects to dicts
        dict_rows = [dict(row) for row in rows]
        
        # Return as a JSON formatted string for the LLM to read easily
        result_str = json.dumps(dict_rows, indent=2)
        
        # Safety limit to prevent exploding the LLM context size
        if len(result_str) > 3000:
             return json.dumps(dict_rows[:5], indent=2) + "\n... (Results truncated due to length. Please refine your query with LIMIT or WHERE)."
             
        return result_str
        
    except sqlite3.OperationalError as e:
         # M2 Pattern: Explicitly return the syntax error to the LLM so it can reflect and fix it
         return f"SQLite Syntax/Operational Error: {str(e)}\nPlease check your query syntax and table names (use get_database_schema if needed) and try again."
    except Exception as e:
         return f"Execution Error: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()

# Export the active tools
tools = [get_database_schema, execute_sql_query]
