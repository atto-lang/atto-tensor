// Host-only verifier tests with synthetic checkouts/runners; NOT native Oreslang.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, appendFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const verifier = fileURLToPath(new URL('../scripts/verify-matmul-vjp-guest.sh', import.meta.url));
function exec(cmd, args, options = {}) {
  return spawnSync(cmd, args, { encoding: 'utf8', timeout: 5000, ...options });
}
function git(cwd, ...args) {
  const r = exec('git', args, { cwd });
  assert.equal(r.status, 0, r.stderr);
  return r.stdout.trim();
}
function makeFixture(t, guestScript) {
  const tmp = mkdtempSync(join(tmpdir(), 'ores-matmul-pinned-'));
  t.after(() => rmSync(tmp, { recursive: true, force: true }));
  const root = join(tmp, 'tensor');
  for (const path of ['scripts', 'src', 'tests']) mkdirSync(join(root, path), { recursive: true });
  copyFileSync(verifier, join(root, 'scripts/verify-matmul-vjp-guest.sh'));
  for (const path of ['src/tensor.ores', 'src/matmul_vjp.ores', 'tests/matmul_vjp_smoke.ores']) {
    writeFileSync(join(root, path), '// synthetic bytes only\n');
  }
  git(root, 'init', '-q');
  git(root, 'config', 'user.name', 'Synthetic');
  git(root, 'config', 'user.email', 'synthetic@example.invalid');
  git(root, 'add', '.');
  git(root, 'commit', '-qm', 'synthetic pinned-tree fixture');
  const runner = join(tmp, 'synthetic-guest.sh');
  writeFileSync(runner, guestScript);
  chmodSync(runner, 0o700);
  const env = {
    ...process.env,
    ORESLANG_TENSOR_EXPECTED_SHA: git(root, 'rev-parse', 'HEAD'),
    ORESLANG_RUNNER_SHA256: createHash('sha256').update(readFileSync(runner)).digest('hex'),
    ORESLANG_GUEST_TIMEOUT_SEC: '1',
  };
  const verify = (updates = {}) =>
    exec('bash', [join(root, 'scripts/verify-matmul-vjp-guest.sh'), runner], {
      cwd: root, env: { ...env, ...updates },
    });
  return { root, env, verify };
}

test('pinned synthetic matmul marker is accepted as host harness behavior only', (t) => {
  const { verify } = makeFixture(t, '#!/bin/sh\necho ORES_TENSOR_MATMUL_DUAL_VJP_PASS\n');
  const result = verify();
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /ORESLANG_TENSOR_MATMUL_GUEST=PASS/);
});

test('wrong executable digest and malformed timeout fail closed', (t) => {
  const { verify } = makeFixture(t, '#!/bin/sh\necho ORES_TENSOR_MATMUL_DUAL_VJP_PASS\n');
  const digest = verify({ ORESLANG_RUNNER_SHA256: 'f'.repeat(64) });
  assert.notEqual(digest.status, 0);
  assert.match(digest.stderr, /digest mismatch/);
  const timeout = verify({ ORESLANG_GUEST_TIMEOUT_SEC: '999' });
  assert.equal(timeout.status, 78);
  assert.match(timeout.stderr, /invalid bounded guest timeout/);
});

test('exact marker with stderr is rejected', (t) => {
  const { verify } = makeFixture(t, '#!/bin/sh\necho ORES_TENSOR_MATMUL_DUAL_VJP_PASS\necho warning >&2\n');
  const result = verify();
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /UNEXPECTED_STDERR/);
});

test('hung compiler is stopped within the guest watchdog', (t) => {
  const { verify } = makeFixture(t, '#!/bin/sh\nsleep 20\n');
  const result = verify();
  assert.notEqual(result.status, 0);
  assert.equal(result.error, undefined, 'bounded watchdog must end before host timeout');
  assert.match(result.stderr, /ORESLANG_TENSOR_MATMUL_GUEST=TIMEOUT/);
});

test('source changes concealed by assume-unchanged fail the exact blob gate', (t) => {
  const { root, verify } = makeFixture(t, '#!/bin/sh\necho ORES_TENSOR_MATMUL_DUAL_VJP_PASS\n');
  git(root, 'update-index', '--assume-unchanged', 'src/matmul_vjp.ores');
  appendFileSync(join(root, 'src/matmul_vjp.ores'), '// hidden replacement\n');
  assert.equal(git(root, 'status', '--porcelain'), '');
  const result = verify();
  assert.equal(result.status, 78);
  assert.match(result.stderr, /source blob differs from exact HEAD/);
});
