import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def run_migration():
    print("Connecting to PostgreSQL...")
    
    # Connect directly to the database specified in DATABASE_URL
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    print("Creating tables...")

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            email VARCHAR(100) NOT NULL UNIQUE,
            password_hash BYTEA NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Upload history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS upload_history (
            id SERIAL PRIMARY KEY,
            user_id INT NOT NULL,
            filename VARCHAR(255) NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)

    # RFM results table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rfm_results (
            id SERIAL PRIMARY KEY,
            file_id INT NOT NULL,
            customer_id VARCHAR(100) NOT NULL,
            recency INT,
            frequency INT,
            monetary DOUBLE PRECISION,
            cluster INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES upload_history(id) ON DELETE CASCADE
        );
    """)

    # Index
    # In Postgres, CREATE INDEX IF NOT EXISTS is supported in newer versions, 
    # but to be safe we can use distinct naming or checking.
    # Simple way: IF NOT EXISTS is widely supported in recent PG versions.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rfm_file ON rfm_results(file_id);")
    print("Index idx_rfm_file ensured.")

    conn.commit()
    cur.close()
    conn.close()

    print("Migration completed successfully.")


if __name__ == "__main__":
    run_migration()
