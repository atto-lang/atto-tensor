#!/usr/bin/env sh
# Native guest acceptance only: exact output and independent bounded watchdog
# for EACH fixture. Synthetic marker runners do not prove real guest execution.
set -eu

if [ -z "${ORESLANG_BIN:-}" ] || [ ! -f "$ORESLANG_BIN" ] || [ ! -x "$ORESLANG_BIN" ]; then
  echo "UNCONFIGURED: ORESLANG_BIN must be an executable Oreslang runner" >&2
  exit 78
fi
case "${ORESLANG_GUEST_TIMEOUT_SEC:-120}" in
  1|5|10|30|60|120) timeout_sec="${ORESLANG_GUEST_TIMEOUT_SEC:-120}" ;;
  *) echo "UNCONFIGURED: invalid bounded guest timeout" >&2; exit 78 ;;
esac
if ! command -v timeout >/dev/null 2>&1; then
  echo "UNCONFIGURED: GNU timeout watchdog is required" >&2
  exit 78
fi
tmpdir="$(mktemp -d)" || exit 78
trap 'rm -rf "$tmpdir"' EXIT

run_case() {
  fixture="$1"
  marker="$2"
  if timeout -k 2s "${timeout_sec}s" "$ORESLANG_BIN" "$fixture" >"$tmpdir/stdout" 2>"$tmpdir/stderr"; then
    :
  else
    guest_rc="$?"
    if [ "$guest_rc" -eq 124 ] || [ "$guest_rc" -eq 137 ]; then
      echo "GUEST_TIMEOUT: $fixture" >&2
    else
      echo "GUEST_FAILED: $fixture (exit=$guest_rc)" >&2
    fi
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

run_rejection() {
  fixture="$1"
  diagnostic="$2"
  guest_rc=0
  timeout -k 2s "${timeout_sec}s" "$ORESLANG_BIN" "$fixture" >"$tmpdir/stdout" 2>"$tmpdir/stderr" || guest_rc="$?"
  if [ "$guest_rc" -ne 1 ] || [ -s "$tmpdir/stdout" ] || ! grep -Fq "$diagnostic" "$tmpdir/stderr"; then
    echo "GUEST_REJECTION_MISMATCH: $fixture (exit=$guest_rc)" >&2
    cat "$tmpdir/stderr" >&2
    exit 1
  fi
  printf 'GUEST_REJECTION_PASS: %s\n' "$fixture"
}

run_case tests/layout_smoke.ores ATTO_TENSOR_LAYOUT_SMOKE_PASS
run_case tests/smoke.ores ORES_ML_TENSOR_SMOKE_PASS
run_case tests/row_batches_smoke.ores ORES_TENSOR_ROW_BATCH_VJP_PASS
run_case tests/matmul_vjp_smoke.ores ORES_TENSOR_MATMUL_DUAL_VJP_PASS
run_case tests/matmul_two_sided_vjp_smoke.ores ORES_MATMUL_TWO_SIDED_VJP_PASS
run_rejection tests/projected_alias_reject.ores 'mutable projected borrows need path-exclusive loan tracking'
