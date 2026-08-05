import sqlite3
import json
import hashlib
import asyncio
from pathlib import Path

class APICache:
    """Thread-safe SQLite-backed API cache for reliable benchmarking."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        
        # Initialize DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT,
                    prompt TEXT,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
    def _generate_key(self, model: str, prompt: str) -> str:
        """Generate a deterministic hash for the request."""
        payload = f"{model}::{prompt}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
        
    async def get(self, model: str, prompt: str) -> str | None:
        """Retrieve a cached response."""
        key = self._generate_key(model, prompt)
        
        def _fetch():
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT response FROM api_cache WHERE cache_key = ?", (key,))
                row = cur.fetchone()
                return row[0] if row else None
                
        return await asyncio.to_thread(_fetch)
        
    async def set(self, model: str, prompt: str, response: str) -> None:
        """Store a response in the cache."""
        key = self._generate_key(model, prompt)
        
        async with self._lock:
            def _insert():
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO api_cache (cache_key, model, prompt, response)
                        VALUES (?, ?, ?, ?)
                        """,
                        (key, model, prompt, response)
                    )
            await asyncio.to_thread(_insert)
