#!/bin/bash
# Start the ISC-Bench Dashboard
cd "$(dirname "$0")"

# Check for venv
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source .venv/bin/activate
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
