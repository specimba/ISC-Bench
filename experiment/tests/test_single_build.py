import pytest
from pathlib import Path
import json
import importlib.util
import sys

# Load single build dynamically
single_build_path = Path(__file__).parent.parent / "isc_single" / "build.py"
spec = importlib.util.spec_from_file_location("single_build", single_build_path)
single_build = importlib.util.module_from_spec(spec)
sys.modules["single_build"] = single_build
spec.loader.exec_module(single_build)

def test_load_queries_txt(tmp_path):
    txt_file = tmp_path / "queries.txt"
    txt_file.write_text("query1\nquery2\n")
    queries = single_build.load_queries(str(txt_file))
    assert len(queries) == 2
    assert queries[0]["query"] == "query1"
    assert queries[1]["query"] == "query2"

def test_load_queries_json(tmp_path):
    json_file = tmp_path / "queries.json"
    json_file.write_text(json.dumps([{"query": "q1", "category": "cat1"}, {"query": "q2"}]))
    queries = single_build.load_queries(str(json_file))
    assert len(queries) == 2
    assert queries[0]["query"] == "q1"
    assert queries[0]["category"] == "cat1"

def test_get_fewshot_examples():
    corpus = {
        "models": ["modelA"],
        "examples": {
            "modelA": {
                "cat1": {"query": "q1", "safe": "s1", "unsafe": "u1"},
                "cat2": {"query": "q2", "safe": "s2", "unsafe": "u2"},
                "cat3": {"query": "q3", "safe": "s3", "unsafe": "u3"}
            }
        }
    }
    
    res = single_build.get_fewshot_examples(corpus, "cat1", 1)
    assert len(res) == 1
    assert res[0]["query"] == "q1"
    
    res = single_build.get_fewshot_examples(corpus, "cat_missing", 1)
    assert len(res) == 1
    
    res = single_build.get_fewshot_examples(corpus, "cat1", 0)
    assert len(res) == 0

def test_build_test_cases():
    fewshot = [{"query": "q_ex", "safe": "s_ex", "unsafe": "u_ex"}]
    
    out_guard = single_build.build_test_cases("my_query", "ai-guard", fewshot)
    parsed_guard = json.loads(out_guard)
    assert len(parsed_guard) == 2 
    assert parsed_guard[0]["query"] == "q_ex"
    assert parsed_guard[1]["query"] == "my_query"
    assert parsed_guard[1]["assistant_safe"] == "???"
    
    out_detoxify = single_build.build_test_cases("my_query", "ai-detoxify", fewshot)
    parsed_detoxify = json.loads(out_detoxify)
    assert len(parsed_detoxify) == 2
    assert parsed_detoxify[0]["prompt"] == "q_ex"
    assert parsed_detoxify[1]["prompt"] == "my_query"
    assert parsed_detoxify[1]["not_flagged"] == "???"
