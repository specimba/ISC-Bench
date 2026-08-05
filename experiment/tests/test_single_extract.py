import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "isc_single"))
import extract

def test_detect_template():
    assert extract.detect_template(Path("some/path/ai-guard/0sample.json")) == "ai-guard"
    assert extract.detect_template(Path("some/path/ai-detoxify/0sample.json")) == "ai-detoxify"
    assert extract.detect_template(Path("some/path/ai-outlier/0sample.json")) == "ai-outlier"
    assert extract.detect_template(Path("some/path/unknown/0sample.json")) == "ai-guard"
