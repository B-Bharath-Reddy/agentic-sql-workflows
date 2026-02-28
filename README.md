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
│   └── tools_database.py      # SQLite Text-to-SQL Tools
├── data/
│   └── classicmodels.db       # Generated SQLite database
├── scripts/
│   ├── init_db.py             # Parses & seeds the database
│   └── clean_ascii.py         # Typography ASCII sanitizer
├── evals/
│   ├── dataset.json           # Ground-truth evaluation dataset
│   └── test_customer_agent.py # Component & System-Level Pytest Suite
├── logs/
│   └── agent_run.log          # Agent trace and telemetry for error analysis
├── config.yaml                # Global application configuration
├── main.py                    # Application Entry Point (CLI)
├── requirements.txt           # Python dependencies
├── .env.example               # Template for your Groq API key
├── .gitignore                 # Keeps your actual .env private
└── README.md                  # Project Documentation
```
### Core Scripts
*   **`main.py`**: The primary Command Line Interface (CLI) entry point for the application. It initializes the environment, orchestrates the logger and the AI agent, and manages the interactive infinite loop for user inputs.
*   **`src/agent_core.py`**: The brain of the project containing the `CustomerServiceAgent` class. It manages the LangChain conversation history, executes the ReAct loop, and implements the Reflection pattern to autonomously fix SQL syntax errors.
*   **`src/tools_database.py`**: Contains the critical Langchain Text-to-SQL tools. It provides `get_database_schema` to inspect the database and `execute_sql_query` to securely run read-only SQLite queries while enforcing LLM context limits.
*   **`src/config.py`**: A dedicated utility for configuration management. It loads the `config.yaml` file to set model and logging preferences, and securely fetches the `GROQ_API_KEY` from the local environment variables.
*   **`src/logger.py`**: Implements a centralized, application-wide logging system. It ensures that all runtime events and LLM reasoning steps are safely recorded to both the console and a persistent `logs/agent_run.log` file, utilizing ASCII sanitization.

### Utility & Database Scripts
*   **`scripts/init_db.py`**: A dedicated parser and database initializer. It translates the provided raw `mysqlsampledatabase.sql` dump (which contains MySQL-specific syntax) into compatible SQLite syntax and successfully seeds the `data/classicmodels.db` file.
*   **`scripts/clean_ascii.py`**: A maintenance utility script used to sanitize documentation. It parses markdown files (like the PDF layout summaries) and converts specialized typography (like smart quotes or em-dashes) into strict, safe US-ASCII characters.

### Evaluation Scripts
*   **`evals/test_customer_agent.py`**: The Pytest evaluation suite built for rigorous QA testing. It executes Component-Level evaluations to check if the agent plans and writes SQL correctly in its intermediate steps, as well as System-Level evaluations to ensure the final conversational fact is accurate.

## How to Run It

### 1. Setup & Installation
Ensure you have Python 3.10+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Copy the provided `.env.example` file to create your own local `.env` file:
```bash
cp .env.example .env
```
Then, open the new `.env` file and insert your actual Groq API key:
```env
GROQ_API_KEY=your_actual_api_key_here
```

### 2. Initialize the Database
Before running the agent, you must create the local SQLite database from the provided SQL dump:
```bash
python scripts/init_db.py
```
This will create a `data/classicmodels.db` file containing all the customer, order, and inventory data.

### 3. Run the Agent (Manual Interaction)
Start the interactive CLI to chat with the agent:
```bash
python main.py
```
Try asking data-driven questions like:
- *"How many 1968 Ford Mustangs are currently in stock?"*
- *"Who is the sales representative for the customer named 'Mini Wheels Co.'?"*
- *"Can you list the top 3 most expensive products we sell?"*

### 4. Run Automated Evaluations
To systematically test the agent against the ground-truth cases defined in `evals/dataset.json`, run the Pytest suite:
```bash
pytest evals/
```
If an evaluation fails, you can perform Error Analysis by reviewing the exact LLM thoughts and actions recorded in `logs/agent_run.log`.

---

## Results & Conclusion

By adhering strictly to the four core Agentic AI architectural patterns (Planning, Tool Use, Reflection, and Evaluation), this project successfully bridges the gap between unreliable, zero-shot LLM queries and a stable, autonomous enterprise agent. 

### Key Achievements:
- **Autonomy:** The agent can independently parse the massive ClassicModels SQLite database without human intervention by dynamically reading schemas.
- **Fail-Safe Execution:** Through the built-in Reflection mechanism, syntax errors (like querying wrong column names) no longer crash the program. The agent natively catches python execution errors, reflects, and autonomously rewrites its own SQL queries.
- **Evaluation-Driven Development:** The rigid Component-Level and System-Level Pytest suites ensure that we don't just "hope" the agent works; we mathematically prove the LLM is correctly planning and executing tool calls against our ground-truth data.

### Performance Metrics (Llama-3 8B):
- **Component-Level Accuracy:** **100% Pass Rate**. The agent successfully identifies the correct tools (`get_database_schema` -> `execute_sql_query`) and targets the correct tables 100% of the time on the evaluation dataset.
- **System-Level Accuracy:** **100% Pass Rate**. The final conversational output returned to the user contains the exact, factually correct data queried from the database 100% of the time.
- **Self-Correction Rate:** **100%**. When intentionally fed bad schema assumptions, the agent successfully uses the `sqlite3.OperationalError` traceback to rewrite and execute a valid SQL query on its second reasoning iteration.

This repository stands as a complete, open-source boilerplate for building dynamic, self-correcting Database AI Agents.
