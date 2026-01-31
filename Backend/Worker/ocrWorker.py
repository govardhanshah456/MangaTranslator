import psycopg2
import time
import os
import sys
from pathlib import Path

# Validate required environment variables
required_env_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "OUTPUT_DIR"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT"))
)

def fetch_and_lock_job(cur):
    cur.execute("""
        UPDATE jobs
        SET status = 'PROCESSING'
        WHERE id = (
            SELECT id FROM jobs
            WHERE status = 'PENDING'
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, document_id;
    """)
    return cur.fetchone()

def process_job(job_id, document_id):
    time.sleep(2)

    output_dir = Path(os.getenv("OUTPUT_DIR"))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{job_id}_output.txt"
    output_path.write_text("Fake translated content")

    return str(output_path)

while True:
    with conn:
        with conn.cursor() as cur:
            job = fetch_and_lock_job(cur)

            if not job:
                print("No job found")
                time.sleep(2)
                continue

            job_id, document_id = job

            try:
                output_path = process_job(job_id, document_id)
                cur.execute("""
                    UPDATE jobs
                    SET status = 'DONE',
                        progress = 100,
                        output_path = %s
                    WHERE id = %s
                """, (output_path, job_id))
            except Exception as e:
                cur.execute("""
                    UPDATE jobs
                    SET status = 'FAILED',
                        error_message = %s
                    WHERE id = %s
                """, (str(e), job_id))