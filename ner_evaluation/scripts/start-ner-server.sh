#!/usr/bin/env bash
# start-ner-server.sh
# ─────────────────────────────────────────────────────────────────────────────
# Starts the NER model server in a separate process.
# Heavy models (HunFlair) load here ONCE and stay in memory.
# The main pipeline calls http://localhost:8001 instead of loading them locally.
#
# Usage:
#   ./start-ner-server.sh                  # start with default settings
#   HUNFLAIR_ENABLED=true ./start-ner-server.sh   # enable HunFlair too
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# Load .env so the server sees the same config as the pipeline
if [ -f ".env" ]; then
    set -a
    source ".env"
    set +a
fi

PORT="${NER_SERVER_PORT:-8001}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Bio-Semantic NER Server"
echo "  URL : http://localhost:${PORT}"
echo "  HF NER    : ${HF_NER_ENABLED:-true}"
echo "  HunFlair  : ${HUNFLAIR_ENABLED:-false}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Models will load in the background. Check /health to see when ready."
echo "  Leave this terminal open. Run your pipeline in another terminal."
echo ""

.venv/bin/python ner_server/server.py
