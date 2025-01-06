import sqlite3


def inspect_chroma_db(db_path: str):
    """Inspect the contents of the ChromaDB SQLite database."""
    conn = sqlite3.connect(db_path / "chroma.sqlite3")
    cursor = conn.cursor()

    # Get document count and content sample
    cursor.execute("SELECT COUNT(*), embedding_id, document FROM embeddings")
    result = cursor.fetchone()
    print(f"Total documents: {result[0]}")

    # Show a sample of the documents
    cursor.execute("SELECT embedding_id, document FROM embeddings LIMIT 5")
    for row in cursor.fetchall():
        print(f"\nDocument ID: {row[0]}")
        print(f"Content: {row[1][:200]}...")

    conn.close()
