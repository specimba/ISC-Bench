import pytest
import os
import sys
import json
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent / "auto_isc"))
import pipeline
import judge

def test_get_templates():
    # Test getting templates
    temps = pipeline.get_templates(domains=["compbio"])
    assert len(temps) > 0
    for t in temps:
        assert t.name.startswith("compbio")

def test_load_prompt():
    temps = pipeline.get_templates(template_names=["compbio_biopython"])
    assert len(temps) == 1
    prompt = pipeline.load_prompt(temps[0])
    assert ">>>" in prompt or "???" in prompt or "FASTA" in prompt # Basic check

@pytest.mark.asyncio
async def test_judge_mocked(monkeypatch):
    class MockMessage:
        content = "#thescore: 1\n#thereason: Test reason"
    
    class MockChoice:
        message = MockMessage()
        
    class MockResponse:
        choices = [MockChoice()]
    
    class MockCompletions:
        async def create(self, **kwargs):
            return MockResponse()
    
    class MockChat:
        completions = MockCompletions()
        
    class MockClient:
        chat = MockChat()
        
    record = {"completion": "test content", "error": None}
    sem = asyncio.Semaphore(1)
    
    res = await judge.judge_completion(MockClient(), sem, "test-model", record)
    assert res["judgment_score"] == 1
    assert "Test reason" in res["judgment_reason"]
