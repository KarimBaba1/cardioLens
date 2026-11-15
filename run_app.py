"""Start CardioLens on Windows, macOS or Linux.

    python run_app.py          (use the project's virtual-environment Python)

1. Retrains the app model from the shared training split (~1 s, deterministic),
   so it always matches this computer's scikit-learn version.
2. Starts the local LLM (Ollama) if it is installed but not running, and tells
   you if the summary model still needs downloading.
3. Starts the web app at http://127.0.0.1:8000
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLLAMA = "http://127.0.0.1:11434"
LLM_MODEL = os.environ.get("CARDIOLENS_LLM", "qwen2.5:7b")


def ollama_up() -> bool:
    try:
        urllib.request.urlopen(OLLAMA + "/api/version", timeout=2)
        return True
    except OSError:
        return False


def find_ollama() -> str | None:
    local = Path.home() / "ollama-local" / "bin_root" / "bin" / "ollama"
    return shutil.which("ollama") or (str(local) if local.exists() else None)


def start_llm() -> None:
    if not ollama_up():
        exe = find_ollama()
        if exe is None:
            print("Local LLM: Ollama not installed -> summaries use the template fallback.\n"
                  "           Install from https://ollama.com then run: ollama pull " + LLM_MODEL)
            return
        env = dict(os.environ)
        local_models = Path.home() / "ollama-local" / "models"
        if "ollama-local" in exe and local_models.exists():
            env.setdefault("OLLAMA_MODELS", str(local_models))
        print("Local LLM: starting Ollama...")
        subprocess.Popen([exe, "serve"], env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        for _ in range(20):
            if ollama_up():
                break
            time.sleep(0.5)
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except OSError:
        print("Local LLM: Ollama did not start -> template fallback.")
        return
    if LLM_MODEL in names:
        print(f"Local LLM: ready ({LLM_MODEL}).")
    else:
        print(f"Local LLM: model missing. Run once:  ollama pull {LLM_MODEL}\n"
              "           (until then summaries use the template fallback)")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # show status lines immediately
    os.chdir(ROOT)
    print("Model: training the app model from data/processed/train.csv ...")
    subprocess.run([sys.executable, "src/train_final_model.py"], check=True,
                   stdout=subprocess.DEVNULL)
    start_llm()
    print("\nCardioLens: http://127.0.0.1:8000   (Ctrl+C to stop)\n")
    subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app",
                    "--host", "127.0.0.1", "--port", "8000"])


if __name__ == "__main__":
    main()
