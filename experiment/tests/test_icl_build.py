import pytest
from pathlib import Path
import json
import importlib.util
import sys

# Load icl build dynamically
icl_build_path = Path(__file__).parent.parent / "isc_icl" / "build.py"
spec = importlib.util.spec_from_file_location("icl_build", icl_build_path)
icl_build = importlib.util.module_from_spec(spec)
sys.modules["icl_build"] = icl_build
spec.loader.exec_module(icl_build)

def test_replace_target_query():
    msg = '''Some text
```json
[
  {"id": 0, "query": "q1", "assistant_safe": "s1", "assistant_unsafe": "u1"},
  {"id": 1, "query": "q2", "assistant_safe": "???", "assistant_unsafe": "???"}
]
```
more text'''
    
    new_msg = icl_build.replace_target_query(msg, "my_new_query")
    assert "my_new_query" in new_msg
    assert "q1" in new_msg
    
    msg_detoxify = '''```json
[
  {"id": 0, "prompt": "p1", "not_flagged": "s1", "flagged": "u1"},
  {"id": 1, "prompt": "p2", "not_flagged": "???", "flagged": "???"}
]
```'''
    new_msg_detoxify = icl_build.replace_target_query(msg_detoxify, "my_detoxify")
    assert "my_detoxify" in new_msg_detoxify
    assert "p1" in new_msg_detoxify

def test_replace_target_query_no_json():
    msg = "No json here"
    with pytest.raises(ValueError, match="No JSON block found"):
        icl_build.replace_target_query(msg, "new")

def test_replace_target_query_no_placeholders():
    msg = '''```json
[
  {"id": 0, "query": "q1", "assistant_safe": "s1", "assistant_unsafe": "u1"}
]
```'''
    with pytest.raises(AssertionError, match="Expected \\?\\?\\?"):
        icl_build.replace_target_query(msg, "new")
