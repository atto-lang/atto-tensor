#!/usr/bin/env bash
# Host acceptance gate: requires a pinned runner and exact clean source.
# Does not independently authenticate compiler provenance or mirror source.
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
  if [[ -L "$root/$path" ]]; then
    echo "tensor guest source must not be a symlink: $path" >&2
    exit 78
  fi
  # Git assume-unchanged and skip-worktree hints can conceal modifications.
  expected_blob="$(git -C "$root" rev-parse "$actual:$path")" || exit 78
  current_blob="$(git -C "$root" hash-object -- "$root/$path")" || exit 78
  if [[ "$expected_blob" != "$current_blob" ]]; then
    echo "tensor guest source blob differs from exact HEAD: $path" >&2
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
case "${ORESLANG_GUEST_TIMEOUT_SEC:-120}" in
  1|5|10|30|60|120) timeout_sec="${ORESLANG_GUEST_TIMEOUT_SEC:-120}" ;;
  *) echo "invalid bounded guest timeout" >&2; exit 78 ;;
esac
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout watchdog is required" >&2
  exit 78
fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf 'ORES_TENSOR_MATMUL_DUAL_VJP_PASS\n' > "$tmp/expected"
if timeout -k 2s "${timeout_sec}s" "$@" "$root/tests/matmul_vjp_smoke.ores" > "$tmp/stdout" 2> "$tmp/stderr"; then
  :
else
  exit_status="$?"
  cat "$tmp/stderr" >&2
  if [[ "$exit_status" == 124 || "$exit_status" == 137 ]]; then
    echo "ORESLANG_TENSOR_MATMUL_GUEST=TIMEOUT" >&2
  fi
  echo "ORESLANG_TENSOR_MATMUL_GUEST=FAILED" >&2
  exit 1
fi
if ! diff -u "$tmp/expected" "$tmp/stdout"; then
  cat "$tmp/stderr" >&2
  echo "ORESLANG_TENSOR_MATMUL_GUEST=OUTPUT_MISMATCH" >&2
  exit 1
fi
if [[ -s "$tmp/stderr" ]]; then
  cat "$tmp/stderr" >&2
  echo "ORESLANG_TENSOR_MATMUL_GUEST=UNEXPECTED_STDERR" >&2
  exit 1
fi
echo "ORESLANG_TENSOR_MATMUL_GUEST=PASS"
