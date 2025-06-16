import sqlite3
import pandas as pd
import os

# File paths relative to this script
csv_files = {
    "users": "users.csv",
    "therapists": "therapists.csv",
    "therapistPatient": "therapistPatient.csv",
    "mood": "mood.csv"
}

# Database file
db_file = "global.db"

# Connect to the SQLite database (creates it if not exists)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Function to create table and import data
def create_and_import(table_name, csv_path):
    try:
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"\n✅ Table created: {table_name}")
        print(df)  # Show all data
    except Exception as e:
        print(f"❌ Failed to load {table_name}: {e}")

# Load each CSV into the corresponding table
for table, path in csv_files.items():
    if os.path.exists(path):
        create_and_import(table, path)
    else:
        print(f"❌ File not found: {path}")

# Show all tables in the database
print("\n📋 Tables in the database:")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for t in tables:
    print(f" - {t[0]}")

# Close DB connection
conn.close()
