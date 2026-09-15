#!/usr/bin/env bash
# In cascade mode, fetch the s2e checkpoint in the background so the service comes
# up immediately on LingMess. Cascade activates automatically once the download
# lands (the resolver re-checks on each request).
set -e

if [ "${COREF_MODEL}" = "cascade" ] && [ "${S2E_AUTO_DOWNLOAD:-true}" != "false" ] \
   && [ ! -f "${S2E_MODEL_PATH}/pytorch_model.bin" ]; then
  echo "[entrypoint] cascade: downloading s2e checkpoint in background; serving LingMess until it's ready."
  ( bash /app/scripts/download_s2e.sh "${S2E_MODEL_PATH}" \
      && echo "[entrypoint] s2e checkpoint ready; cascade activates on the next request." \
      || echo "[entrypoint] s2e download failed; staying LingMess-only." ) &
fi

exec "$@"
