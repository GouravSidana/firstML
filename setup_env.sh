#!/usr/bin/env bash
set -e
if [ -d ".venv" ]; then
  echo ".venv already exists. Activate with 'source .venv/bin/activate' or remove .venv to recreate."
  exit 0
fi
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Done. Activate with: source .venv/bin/activate"
