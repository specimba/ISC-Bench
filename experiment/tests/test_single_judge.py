import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "isc_single"))
import judge

def test_extract_content():
    text = "Blah blah\n#thescore: 1\n#thereason: This is harmful."
    score = judge.extract_content("#thescore:", text)
    assert score == 1
    
    reason = judge.extract_content("#thereason:", text)
    assert reason == "This is harmful."
    
    text_invalid = "#thescore: invalid"
    assert judge.extract_content("#thescore:", text_invalid) is None
    
    assert judge.extract_content("#missing:", text) is None
