#!/usr/bin/env bash
# Start CardioLens: the local LLM (Ollama, if installed) and the web app.
# Usage: ./run_app.sh            then open http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"

OLLAMA_BIN="${OLLAMA_BIN:-$HOME/ollama-local/bin_root/bin/ollama}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/ollama-local/models}"
export OLLAMA_HOST="127.0.0.1:11434"

if curl -s --max-time 2 "http://$OLLAMA_HOST/api/version" >/dev/null; then
  echo "Local LLM: Ollama already running."
elif [ -x "$OLLAMA_BIN" ] || command -v ollama >/dev/null; then
  BIN="$OLLAMA_BIN"; [ -x "$BIN" ] || BIN="$(command -v ollama)"
  echo "Local LLM: starting Ollama..."
  nohup "$BIN" serve > /tmp/cardiolens-ollama.log 2>&1 &
  sleep 3
else
  echo "Local LLM: Ollama not found - summaries will use the template fallback."
fi

[ -f models/logistic_regression.joblib ] || .venv/bin/python src/train_final_model.py
echo "CardioLens: http://127.0.0.1:8000   (Ctrl+C to stop)"
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
