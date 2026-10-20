#!/usr/bin/env bash
# Certifying guest entrypoint only when invoked with a REAL pinned Oreslang
# compiler and an exact clean tensor repo worktree. Does not mirror anything.
set -euo pipefail
if (($# == 0)); then
  echo "usage: $0 /absolute/path/to/pinned/oreslang [runner-args...]" >&2
  exit 64
fi
if [[ -z "${ORESLANG_TENSOR_EXPECTED_SHA:-}" || -z "${ORESLANG_RUNNER_SHA256:-}" ]]; then
  echo "must pin ORESLANG_TENSOR_EXPECTED_SHA and ORESLANG_RUNNER_SHA256" >&2
  exit 64
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -e "$root/.git" ]]; then
  echo "not an independent Git tensor checkout" >&2
  exit 78
fi
actual="$(git -C "$root" rev-parse HEAD)"
if [[ "$actual" != "$ORESLANG_TENSOR_EXPECTED_SHA" ]]; then
  echo "Oreslang tensor Git HEAD mismatches pinned source" >&2
  exit 78
fi
if [[ -n "$(git -C "$root" status --porcelain --untracked-files=all)" ]]; then
  echo "tensor repository contains modified or untracked source" >&2
  exit 78
fi
for path in src/tensor.ores src/matmul_vjp.ores tests/matmul_vjp_smoke.ores; do
  if [[ ! -f "$root/$path" ]]; then
    echo "missing expected tensor source $path" >&2
    exit 78
  fi
  if ! git -C "$root" ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    echo "tensor guest source is untracked or ignored: $path" >&2
    exit 78
  fi
done
runner="$1"
if [[ "$runner" != /* || ! -f "$runner" || ! -x "$runner" ]]; then
  echo "guest compiler must be an absolute executable file" >&2
  exit 64
fi
if command -v sha256sum >/dev/null 2>&1; then
  runner_hash="$(sha256sum "$runner")"; runner_hash="${runner_hash%% *}"
elif command -v shasum >/dev/null 2>&1; then
  runner_hash="$(shasum -a 256 "$runner")"; runner_hash="${runner_hash%% *}"
else
  echo "SHA-256 utility unavailable" >&2
  exit 78
fi
if [[ "$runner_hash" != "$ORESLANG_RUNNER_SHA256" ]]; then
  echo "Oreslang compiler executable digest mismatch" >&2
  exit 78
fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf 'ORES_TENSOR_MATMUL_DUAL_VJP_PASS\n' > "$tmp/expected"
if ! "$@" "$root/tests/matmul_vjp_smoke.ores" > "$tmp/stdout" 2> "$tmp/stderr"; then
  cat "$tmp/stderr" >&2
  echo "ORESLANG_TENSOR_MATMUL_GUEST=FAILED" >&2
  exit 1
fi
if ! diff -u "$tmp/expected" "$tmp/stdout"; then
  cat "$tmp/stderr" >&2
  echo "ORESLANG_TENSOR_MATMUL_GUEST=OUTPUT_MISMATCH" >&2
  exit 1
fi
echo "ORESLANG_TENSOR_MATMUL_GUEST=PASS"
