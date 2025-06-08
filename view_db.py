import sqlite3

def view_database(db_path='global.db'):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            print("No tables found in the database.")
            return

        print(f"Tables in '{db_path}':")
        for table_name in tables:
            table = table_name[0]
            print(f"\nTable: {table}")
            cursor.execute(f"SELECT * FROM {table} LIMIT 10;")
            rows = cursor.fetchall()

            # Get column names
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [info[1] for info in cursor.fetchall()]
            
            # Print column headers
            print(" | ".join(columns))
            print("-" * (len(" | ".join(columns)) + 10))

            # Print rows
            for row in rows:
                print(" | ".join(str(item) for item in row))

        conn.close()
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    view_database()