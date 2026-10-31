#!/usr/bin/env bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
echo "Setup done. Next: python -m src.train  then  streamlit run app.py"
