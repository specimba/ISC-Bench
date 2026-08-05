import pytest
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))

from core.cache import APICache
from core.reporter import BenchmarkReporter

@pytest.mark.asyncio
async def test_api_cache(tmp_path):
    db_path = tmp_path / "test_cache.db"
    cache = APICache(db_path)
    
    await cache.set("test-model", "hello", "world")
    
    val = await cache.get("test-model", "hello")
    assert val == "world"
    
    val_missing = await cache.get("test-model", "missing")
    assert val_missing is None
    
def test_reporter(tmp_path):
    results = [
        {"model": "gpt-4", "score": 1},
        {"model": "gpt-4", "score": 0},
        {"model": "claude", "score": 1},
        {"model": "claude", "score": 1}
    ]
    
    out_dir = tmp_path / "reports"
    rep = BenchmarkReporter(results, out_dir)
    rep.generate_markdown()
    rep.generate_json()
    
    assert (out_dir / "results.json").exists()
    assert (out_dir / "leaderboard.md").exists()
    
    md = (out_dir / "leaderboard.md").read_text()
    assert "gpt-4" in md
    assert "claude" in md
    assert "50.0%" in md # 1/2 for gpt-4
    assert "100.0%" in md # 2/2 for claude
