import pytest
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent / "isc_agent"))
import agent

def test_build_tvd_prompt():
    prompt = agent.build_tvd_prompt(Path("/tmp/workspace"))
    assert "/tmp/workspace" in prompt
    assert "autonomous agent" in prompt
    assert "validator passes" in prompt

def test_run_shell_mocked(tmp_path):
    cwd = tmp_path
    res = agent.run_shell("echo 'hello agent'", cwd)
    assert "hello agent" in res

def test_build_openrouter_model_no_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        agent.build_openrouter_model("test-model")

def test_build_openrouter_model_with_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = agent.build_openrouter_model("test-model", thinking=True)
    # verify thinking setup
    assert "x-or-reasoning" in [k.lower() for k in model._client.default_headers.keys()]

def test_build_anthropic_model_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        agent.build_anthropic_model("test-model")
