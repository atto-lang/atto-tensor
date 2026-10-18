#!/usr/bin/env sh
# Explicit binary only: no fake compiler or silent fallback.
set -eu
if [ -z "${ORESLANG_BIN:-}" ] || [ ! -x "$ORESLANG_BIN" ]; then
  echo "UNCONFIGURED: set ORESLANG_BIN to a real executable Oreslang compiler/runner" >&2
  exit 78
fi
"$ORESLANG_BIN" tests/smoke.ores
"$ORESLANG_BIN" tests/row_batches_smoke.ores
