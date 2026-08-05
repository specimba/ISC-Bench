# Experiment Framework Tests

This directory contains `pytest` unit tests for the Python scripts in `isc_agent`, `isc_single`, and `isc_icl`.

## How to run

These tests require `pytest` and the main experiment dependencies.

```bash
# from the ISC-Bench root
python3 -m venv .venv
source .venv/bin/activate
pip install pytest pytest-asyncio openai python-dotenv tqdm docker openai-agents rich

pytest experiment/tests/ -v
```
