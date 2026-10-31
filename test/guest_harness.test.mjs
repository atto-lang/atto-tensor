// Harness self-tests only: synthetic executables never certify Oreslang execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { chmodSync, existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
function check(executable) {
  return spawnSync('sh', ['scripts/test.sh'], {
    cwd: root,
    env: { ...process.env, ORESLANG_BIN: executable },
    encoding: 'utf8',
    timeout: 10000,
  });
}
function fakeRunner(t, script) {
  const dir = mkdtempSync(join(tmpdir(), 'ores-tensor-harness-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const executable = join(dir, 'fake-guest.sh');
  writeFileSync(executable, script);
  chmodSync(executable, 0o700);
  return executable;
}

test('a silent successful executable cannot pass guest smoke tests', { skip: !existsSync('/bin/true') }, () => {
  const result = check('/bin/true');
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /GUEST_OUTPUT_MISMATCH/);
});

test('every synthetic fixture must emit its distinct exact marker', (t) => {
  const runner = fakeRunner(t, `#!/bin/sh
case "$1" in
  tests/smoke.ores) echo ORES_ML_TENSOR_SMOKE_PASS;;
  tests/row_batches_smoke.ores) echo ORES_TENSOR_ROW_BATCH_VJP_PASS;;
  tests/matmul_vjp_smoke.ores) echo ORES_TENSOR_MATMUL_DUAL_VJP_PASS;;
  tests/matmul_two_sided_vjp_smoke.ores) echo ORES_MATMUL_TWO_SIDED_VJP_PASS;;
  *) exit 23;;
esac
`);
  assert.equal(check(runner).status, 0);
  const wrong = fakeRunner(t, '#!/bin/sh\necho UNRELATED_SUCCESS\n');
  const rejected = check(wrong);
  assert.notEqual(rejected.status, 0);
  assert.match(rejected.stderr, /GUEST_OUTPUT_MISMATCH/);
});

test('unexpected stderr rejects an otherwise matching marker', (t) => {
  const runner = fakeRunner(t, '#!/bin/sh\necho ORES_ML_TENSOR_SMOKE_PASS\necho noisy-compiler >&2\n');
  const result = check(runner);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /GUEST_UNEXPECTED_STDERR/);
});
