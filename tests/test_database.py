import os
import sqlite3
import pytest

# Find the real schema file (we assume init_db.py or a schema.sql exists)
# For smoke testing without mutating the real database, we can create an
# in-memory SQLite database and test our connection logic against it.
@pytest.fixture
def mock_db():
    """
    Creates an isolated, in-memory SQLite database for testing to ensure
    we do not accidentally mutate or lock the real Production/Local DB.
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # Create a dummy table to prove the DB spun up correctly
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mock_test_table (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    ''')
    conn.commit()
    
    yield conn
    
    # Teardown
    conn.close()

def test_database_connection_isolation(mock_db):
    """
    Tests that we can successfully connect to a SQLite instance
    and retrieve table structures without touching data/classicmodels.db.
    """
    cursor = mock_db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert "mock_test_table" in tables, "Failed to build isolated testing dataset."

def test_production_db_file_exists():
    """
    A read-only check to simply ensure the 'make setup' command 
    successfully generated the file for the Agent to use later.
    """
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'classicmodels.db')
    assert os.path.exists(db_path), "Production Database file missing. Did you run `make setup`?"
