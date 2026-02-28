import sqlite3
import os

def init_db():
    sql_file = "mysqlsampledatabase.sql"
    db_file = "data/classicmodels.db"
    
    os.makedirs(os.path.dirname(db_file), exist_ok=True)

    if not os.path.exists(sql_file):
        print(f"Error: {sql_file} not found.")
        return

    print("Reading SQL file...")
    with open(sql_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print("Converting MySQL dump to SQLite syntax via line-by-line parsing...")
    
    sqlite_statements = []
    current_statement = ""
    in_insert = False

    for line in lines:
        stripped = line.strip()
        
        # Skip comments and empty lines
        if not stripped or stripped.startswith('/*') or stripped.startswith('*') or stripped.startswith('--'):
            continue
            
        # Skip MySQL specific session settings
        if stripped.startswith('SET ') or stripped.startswith('USE ') or stripped.startswith('DROP TABLE'):
            continue
            
        # Clean Table Creation
        if stripped.startswith('CREATE TABLE'):
             current_statement = stripped + "\n"
             continue
             
        if current_statement.startswith('CREATE TABLE'):
            # Basic type replacements for SQLite compatibility
            clean_line = line
            clean_line = clean_line.replace(' mediumtext', ' TEXT')
            clean_line = clean_line.replace(' mediumblob', ' BLOB')
            clean_line = clean_line.replace(' smallint(6)', ' INTEGER')
            clean_line = clean_line.replace(' int(', ' INTEGER(')
            clean_line = clean_line.replace(' int ', ' INTEGER ')
            clean_line = clean_line.replace(' int,', ' INTEGER,')
            
            # The last line of CREATE TABLE
            if clean_line.strip().startswith(') ENGINE='):
                clean_line = ");\n"
            elif clean_line.strip() == ");":
                pass
            
            # Append line
            current_statement += clean_line
            
            # If end of statement
            if current_statement.strip().endswith(';'):
                sqlite_statements.append(current_statement.strip())
                current_statement = ""
            continue

        # Handle Inserts - MySQL allows bulk inserts with multiple value sets. We need to escape single quotes cleanly.
        if stripped.lower().startswith('insert into') or stripped.lower().startswith('insert  into'):
            current_statement = line
            # It might be a one-liner
            if current_statement.strip().endswith(';'):
                # SQLite escaping
                # Replace \' with ''
                stm = current_statement.replace("\\'", "''")
                sqlite_statements.append(stm.strip())
                current_statement = ""
            continue
            
        # Continuation of an insert statement
        if current_statement.lower().startswith('insert'):
            current_statement += line
            if current_statement.strip().endswith(';'):
                stm = current_statement.replace("\\'", "''")
                sqlite_statements.append(stm.strip())
                current_statement = ""
            continue

    print("Connecting to SQLite database...")
    if os.path.exists(db_file):
        os.remove(db_file)
        
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = OFF;")

    print(f"Executing {len(sqlite_statements)} converted statements...")
    success_count = 0
    error_count = 0
    
    for stmt in sqlite_statements:
        try:
            cursor.execute(stmt)
            success_count += 1
        except sqlite3.Error as e:
            print(f"Error executing statement: {e}")
            print(f"Statement snippet: {stmt[:100]}...")
            error_count += 1

    conn.commit()
    conn.close()
    
    print("\n--- Migration Complete ---")
    print(f"Successfully executed: {success_count} statements")
    if error_count > 0:
        print(f"Failed statements: {error_count}")
        return False
    else:
        print(f"Successfully created database at {db_file} with NO errors!")
        return True

if __name__ == "__main__":
    init_db()
