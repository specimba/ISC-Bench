# Auto-ISC Pipeline

Automated pipeline for generating large-scale harmful content datasets using Internal Safety Collapse (ISC) vulnerabilities across frontier models.

## Architecture

The pipeline consists of three stages:

1. **Pipeline Execution (`pipeline.py`)**: Batches selected TVD templates across multiple models in parallel to elicit outputs.
2. **Auto-Judging (`judge.py`)**: Uses a strong judge model (e.g., GPT-4o) to verify whether the completion successfully executed the workflow and produced actionable harmful content.
3. **Dataset Export**: Collects the successful completions into a curated JSONL dataset.

## Setup

Requires Python 3.11+, `uv` (or pip install requirements), and `.env` configured with `OPENROUTER_API_KEY`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai tqdm python-dotenv aiofiles
```

## Running the Pipeline

Generate content across specific domains or templates:

```bash
# Run across all compbio templates using two models
python pipeline.py --models openai/gpt-4o,anthropic/claude-3.5-sonnet --domains compbio

# Run specific templates
python pipeline.py --templates aiml_moderation,cyber_angr --models meta-llama/llama-3-70b-instruct
```

Evaluate the generated JSONL outputs:

```bash
python judge.py results/auto_isc_output.jsonl
```
