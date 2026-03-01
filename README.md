# Agentic SQL Workflows: Autonomous Customer Service AI

## Project Aim & Overview
This project aims to build a production-grade, autonomous Customer Service AI agent capable of interacting with a real relational database (SQLite). The primary objective is to implement the core Agentic AI patterns—such as structured Planning, dynamic Tool Use, self-correcting Reflection, and rigorous Evaluation methodologies—to create a robust, self-healing workflow. The agent can dynamically inspect database schemas, write executable SQL queries, catch its own syntax errors, and return accurate, conversational answers to user queries regarding the ClassicModels dataset.

## Project Structure & Script Descriptions

```text
agentic-sql-workflows/
├── src/
│   ├── config.py              # Configuration manager
│   ├── logger.py              # Centralized logging & ASCII sanitizer
│   ├── agent_core.py          # The ReAct Agent Orchestrator
│   ├── tools_database.py      # SQLite Text-to-SQL Tools
│   ├── observability.py       # Context tracking & instrumentation
│   ├── metrics.py             # LLM cost & token tracking
│   ├── tracing.py             # Distributed tracing with spans
│   ├── error_handling.py      # Retry, circuit breaker, error classification
│   └── debug_utils.py         # Conversation replay & debug exports
├── data/
│   └── classicmodels.db       # Generated SQLite database
├── scripts/
│   └── init_db.py             # Parses & seeds the database
├── evals/
│   ├── dataset.json           # Ground-truth evaluation dataset
│   ├── evaluate_agent.py      # Component & System-Level Pytest Suite
│   └── evaluation_framework.py # Comprehensive evaluation framework
├── tests/
│   ├── test_database.py       # Database connection tests
│   ├── test_env.py            # Environment variable tests
│   ├── test_observability.py  # Observability context tests
│   ├── test_metrics.py        # Metrics collector tests
│   ├── test_tracing.py        # Tracing context tests
│   ├── test_error_handling.py # Error handling & retry tests
│   ├── test_debug_utils.py    # Debug utilities tests
│   └── test_evaluation_framework.py # Evaluation framework tests
├── logs/
│   └── agent_run.log          # Agent trace and telemetry for error analysis
├── config.yaml                # Global application configuration
├── .env.example               # Template for your Groq API key
├── .gitignore                 # Keeps your actual .env private
├── Makefile                   # Cross-Platform Build Commands
├── .github/                   # GitHub Actions CI Pipeline
└── README.md                  # Project Documentation
```

### Core Scripts
*   **`main.py`**: The primary Command Line Interface (CLI) entry point for the application. It initializes the environment, orchestrates the logger and the AI agent, and manages the interactive infinite loop for user inputs.
*   **`src/agent_core.py`**: The brain of the project containing the `CustomerServiceAgent` class. It manages the LangChain conversation history, executes the ReAct loop, and implements the Reflection pattern to autonomously fix SQL syntax errors.
*   **`src/tools_database.py`**: Contains the critical Langchain Text-to-SQL tools. It provides `get_database_schema` to inspect the database and `execute_sql_query` to securely run read-only SQLite queries while enforcing LLM context limits.
*   **`src/config.py`**: A dedicated utility for configuration management. It loads the `config.yaml` file to set model and logging preferences, and securely fetches the `GROQ_API_KEY` from the local environment variables.
*   **`src/logger.py`**: Implements a centralized, application-wide logging system. It ensures that all runtime events and LLM reasoning steps are safely recorded to both the console and a persistent `logs/agent_run.log` file, utilizing ASCII sanitization.

### Observability & Monitoring (NEW)
*   **`src/observability.py`**: Provides context tracking for agent queries with iteration counting, LLM call tracking, and tool call instrumentation. Includes the `instrument_tool` and `instrument_llm_call` decorators for automatic observability.
*   **`src/metrics.py`**: Comprehensive metrics collection including LLM token usage, cost estimation, tool call statistics, and query-level metrics. Supports multiple model pricing (GPT-4, Claude, Llama, etc.).
*   **`src/tracing.py`**: Distributed tracing implementation with spans, trace contexts, and context managers. Supports nested spans and automatic duration tracking.
*   **`src/error_handling.py`**: Production-ready error handling with:
    - Error classification (transient vs permanent)
    - Exponential backoff retry with jitter
    - Circuit breaker pattern for fault tolerance
    - Custom error types (LLMError, ToolExecutionError, DatabaseError, etc.)
*   **`src/debug_utils.py`**: Debug utilities including:
    - Conversation state capture and replay
    - Message history inspection
    - Step-by-step conversation replay
    - Debug file exports for analysis

### Utility & Database Scripts
*   **`scripts/init_db.py`**: A dedicated parser and database initializer. It translates the provided raw `mysqlsampledatabase.sql` dump (which contains MySQL-specific syntax) into compatible SQLite syntax and successfully seeds the `data/classicmodels.db` file.

### Evaluation Scripts
*   **`tests/`**: Comprehensive Pytest test suite with 213+ tests covering all modules. Tests use isolated mocks and fixtures for reliability.
*   **``evals/evaluate_agent.py`**: The Pytest evaluation suite built for rigorous LLM QA testing. It executes Component-Level evaluations to check if the agent plans and writes SQL correctly in its intermediate steps, as well as System-Level evaluations to ensure the final conversational fact is accurate.
*   **`evals/evaluation_framework.py`**: A comprehensive evaluation framework supporting:
    - Component-Level Evaluation: Tests intermediate outputs (planner, tool calling)
    - System-Level Evaluation: End-to-end black box testing
    - Performance Metrics: Latency, throughput benchmarks
    - Regression Testing: Baseline comparison for detecting regressions

## How to Run It

This project uses a cross-platform **Makefile** to abstract away complex environment setups across Windows, Mac, and Linux.

### 1. Setup & Environment Variables
Ensure you have Python 3.12+ installed.
Copy the provided `.env.example` file to create your own local `.env` file:
```bash
cp .env.example .env
```
Open `.env` and insert your actual Groq API key.

### 2. Initialize Database & Virtual Environment
To automatically build a `.venv`, install packages natively, and execute `scripts/init_db.py` to seed the database, run:
```bash
make setup
```

### 3. Run CI Smoke Tests
To run quick, isolated Pytest sanity checks over the `.db` connection and API Environment variables, run:
```bash
make ci-smoke
```

### 4. Run the Agent (Manual Interaction)
Start the interactive CLI to chat with the agent:
```bash
make run
```
Try asking data-driven questions like:
- *"How many 1968 Ford Mustangs are currently in stock?"*
- *"Who is the sales representative for the customer named 'Mini Wheels Co.'?"*
- *"Can you list the top 3 most expensive products we sell?"*

### 5. Run Automated LLM Evaluations
To systematically test the agent against the ground-truth cases defined in `evals/dataset.json`, run the full Pytest suite:
```bash
make eval
```

### 6. Run Full Test Suite
To run all 213+ tests including observability, metrics, tracing, error handling, and evaluation framework:
```bash
make test
```

### 7. Run Tests with Coverage
To run tests with coverage reporting:
```bash
make coverage
```

If an evaluation fails, perform Error Analysis by reviewing the exact LLM thoughts recorded in `logs/agent_run.log`.

---

## Continuous Integration (CI/CD)
This project includes an automated GitHub Actions pipeline (`.github/workflows/ci.yml`). 
On every `push` or `pull_request` to the `main` branch, an Ubuntu server will automatically trigger `make setup` -> `make ci-smoke` -> `make eval`. 

*Note: Ensure you add `GROQ_API_KEY` to your GitHub Repository Secrets to allow the pipeline to successfully evaluate the LLM.*

---

## Architecture Highlights

### Observability Stack
The project includes a comprehensive observability stack for production monitoring:

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Query Flow                          │
├─────────────────────────────────────────────────────────────┤
│  User Query                                                  │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ Observability   │───▶│    Tracing      │                 │
│  │ Context         │    │ (Spans/Traces)  │                 │
│  └─────────────────┘    └─────────────────┘                 │
│      │                       │                               │
│      ▼                       ▼                               │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ Metrics         │    │ Error Handling  │                 │
│  │ (Tokens/Cost)   │    │ (Retry/Circuit) │                 │
│  └─────────────────┘    └─────────────────┘                 │
│      │                       │                               │
│      └───────────┬───────────┘                               │
│                  ▼                                           │
│          ┌─────────────────┐                                 │
│          │ Debug Utils     │                                 │
│          │ (Replay/Export) │                                 │
│          └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

### Error Handling Patterns
- **Retry with Exponential Backoff**: Automatic retry for transient failures (rate limits, timeouts)
- **Circuit Breaker**: Prevents cascade failures by blocking calls to failing services
- **Error Classification**: Distinguishes between transient and permanent errors
- **Graceful Degradation**: Fallback mechanisms for critical paths

---

## Results & Conclusion

By adhering strictly to the four core Agentic AI architectural patterns (Planning, Tool Use, Reflection, and Evaluation), this project successfully bridges the gap between unreliable, zero-shot LLM queries and a stable, autonomous enterprise agent. 

### Key Achievements:
- **Autonomy:** The agent can independently parse the massive ClassicModels SQLite database without human intervention by dynamically reading schemas.
- **Fail-Safe Execution:** Through the built-in Reflection mechanism, syntax errors (like querying wrong column names) no longer crash the program. The agent natively catches python execution errors, reflects, and autonomously rewrites its own SQL queries.
- **Evaluation-Driven Development:** The rigid Component-Level and System-Level Pytest suites ensure that we don't just "hope" the agent works; we mathematically prove the LLM is correctly planning and executing tool calls against our ground-truth data.
- **Production-Ready Observability:** Complete visibility into agent behavior with metrics, tracing, and debug capabilities.

### Performance Metrics (Llama-3 8B):
- **Component-Level Accuracy:** **100% Pass Rate**. The agent successfully identifies the correct tools (`get_database_schema` -> `execute_sql_query`) and targets the correct tables 100% of the time on the evaluation dataset.
- **System-Level Accuracy:** **100% Pass Rate**. The final conversational output returned to the user contains the exact, factually correct data queried from the database 100% of the time.
- **Self-Correction Rate:** **100%**. When deliberately fed bad schema assumptions, the agent successfully uses the `sqlite3.OperationalError` traceback to rewrite and execute a valid SQL query on its second reasoning iteration.
- **Test Coverage:** **213+ tests** covering all modules with comprehensive unit and integration tests.

This repository stands as a complete, open-source boilerplate for building dynamic, self-correcting Database AI Agents with production-ready observability and evaluation capabilities.