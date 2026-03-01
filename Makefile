.PHONY: help setup activate ci-smoke run eval test coverage clean
.DEFAULT_GOAL := help

VENV_DIR ?= .venv
# Use pyvenv.cfg as the venv creation target (exists for both Scripts/ and bin/).
VENV_CFG := $(VENV_DIR)/pyvenv.cfg
# Stamp file ensures deps are reinstalled when requirements.txt changes.
VENV_STAMP := $(VENV_DIR)/.deps_installed
# VENV_BIN is resolved after venv creation; pyvenv.cfg avoids first-run empty targets.

# MSYSTEM can mean Git Bash+Windows CPython (Scripts/) or MSYS2 Python (bin/),
# so we detect layout after venv creation instead of hardcoding.
VENV_BIN = $(if $(wildcard $(VENV_DIR)/Scripts/python.exe),$(VENV_DIR)/Scripts,$(VENV_DIR)/bin)
# Prefer python.exe if present (Windows), otherwise python (Unix/MSYS2).
VENV_PY = $(firstword $(wildcard $(VENV_BIN)/python.exe) $(wildcard $(VENV_BIN)/python))
VENV_ACT = $(VENV_BIN)/activate

ifdef MSYSTEM
	PYTHON ?= python
	VENV_FLAGS ?= --copies
else ifeq ($(OS),Windows_NT)
	PYTHON ?= python
else
	PYTHON ?= python3
endif

help:
	@echo "Available targets:"
	@echo "  setup     Create virtual environment, install dependencies, and init DB"
	@echo "  activate  Show how to activate the virtual environment"
	@echo "  ci-smoke  Run fast checks to ensure python and DB compile correctly"
	@echo "  run       Starts the interactive AI CLI agent"
	@echo "  eval      Run the evaluation suite (Component & System level)"
	@echo "  test      Run all tests including observability, metrics, tracing"
	@echo "  coverage  Run tests with coverage reporting"
	@echo "  clean     Remove build caches and temporary files"
	@echo ""

setup: $(VENV_STAMP)
	@echo "Initializing ClassicModels SQLite database..."
	$(VENV_PY) scripts/init_db.py
	@echo "Environment setup complete. The agent is ready."

$(VENV_CFG):
	$(PYTHON) -m venv $(VENV_FLAGS) $(VENV_DIR)

$(VENV_STAMP): requirements.txt $(VENV_CFG)
	@$(if $(filter Windows_NT,$(OS)),if not exist "$(VENV_PY)" (echo "Venv python not found. Venv creation failed." && exit 1),test -f "$(VENV_PY)" || (echo "Venv python not found. Venv creation failed." && exit 1))
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt
	@echo ok > $(VENV_STAMP)

activate:
	@echo "Run the following to activate your environment:"
ifdef MSYSTEM
	@echo "  source $(VENV_ACT)"
else ifeq ($(OS),Windows_NT)
	@echo "  PowerShell: $(VENV_DIR)\\Scripts\\Activate.ps1"
	@echo "  CMD:        $(VENV_DIR)\\Scripts\\activate.bat"
	@echo "  Note: If PowerShell blocks Activate.ps1, run: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
else
	@echo "  source $(VENV_ACT)"
endif

ci-smoke: $(VENV_STAMP)
	@echo "Running CI Smoke Test..."
	$(VENV_PY) -m pytest -q tests/
	@echo "Smoke test passed. Environment is healthy."

run: $(VENV_STAMP)
	@echo "Starting Agentic Workflow CLI..."
	$(VENV_PY) main.py

eval: $(VENV_STAMP)
	@echo "Running System and Component Evaluations..."
	$(VENV_PY) -m pytest evals/evaluate_agent.py

test: $(VENV_STAMP)
	@echo "Running all tests..."
	$(VENV_PY) -m pytest -v tests/
	@echo "All tests passed."

coverage: $(VENV_STAMP)
	@echo "Running tests with coverage..."
	$(VENV_PY) -m pytest --cov=src --cov-report=term-missing --cov-report=html tests/
	@echo "Coverage report generated in htmlcov/"

clean:
ifdef MSYSTEM
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage $(VENV_DIR) data/classicmodels.db
else ifeq ($(OS),Windows_NT)
	powershell -NoProfile -Command 'Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue; exit 0'
	powershell -NoProfile -Command 'Remove-Item -Recurse -Force .pytest_cache,.mypy_cache,htmlcov,.coverage,$(VENV_DIR),data/classicmodels.db -ErrorAction SilentlyContinue; exit 0'
else
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage $(VENV_DIR) data/classicmodels.db
endif
	@echo "Cleaned derived data, venv, and caches."
