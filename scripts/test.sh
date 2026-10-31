#!/usr/bin/env sh
# Native guest acceptance entrypoint; success requires each real guest marker.
# An executable that silently returns zero (e.g. /bin/true) must fail closed.
set -eu
if [ -z "${ORESLANG_BIN:-}" ] || [ ! -x "$ORESLANG_BIN" ]; then
  echo "UNCONFIGURED: ORESLANG_BIN must be an executable Oreslang runner" >&2
  exit 78
fi
tmpdir="$(mktemp -d)" || exit 78
trap 'rm -rf "$tmpdir"' EXIT

run_case() {
  fixture="$1"
  marker="$2"
  if ! "$ORESLANG_BIN" "$fixture" >"$tmpdir/stdout" 2>"$tmpdir/stderr"; then
    echo "GUEST_FAILED: $fixture" >&2
    cat "$tmpdir/stderr" >&2
    exit 1
  fi
  printf '%s\n' "$marker" >"$tmpdir/expected"
  if ! diff -u "$tmpdir/expected" "$tmpdir/stdout"; then
    echo "GUEST_OUTPUT_MISMATCH: $fixture" >&2
    cat "$tmpdir/stderr" >&2
    exit 1
  fi
  if [ -s "$tmpdir/stderr" ]; then
    echo "GUEST_UNEXPECTED_STDERR: $fixture" >&2
    cat "$tmpdir/stderr" >&2
    exit 1
  fi
  printf 'GUEST_CASE_PASS: %s\n' "$fixture"
}

run_case tests/smoke.ores ORES_ML_TENSOR_SMOKE_PASS
run_case tests/row_batches_smoke.ores ORES_TENSOR_ROW_BATCH_VJP_PASS
run_case tests/matmul_vjp_smoke.ores ORES_TENSOR_MATMUL_DUAL_VJP_PASS
run_case tests/matmul_two_sided_vjp_smoke.ores ORES_MATMUL_TWO_SIDED_VJP_PASS
