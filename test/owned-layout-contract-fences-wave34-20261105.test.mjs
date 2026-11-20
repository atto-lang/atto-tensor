// Source-local contract and reference admission checks, not Oreslang guest execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const c = JSON.parse(readFileSync(new URL('../integration/contract.v1.json', import.meta.url), 'utf8'));

test('layout fixtures are exact versioned leaf IDs, not self-certified blobs', () => {
  assert.deepEqual(c.fixtureIds, ['tensor/add/v1','tensor/matmul/v1','tensor/shape/v1']);
  for(const id of c.fixtureIds) {
    assert.match(id, /^tensor\/[a-z][a-z0-9-]*\/v[1-9][0-9]*$/);
    assert.ok(!id.split('/').some(s => s === '' || s === '.' || s === '..'));
  }
});

test('CPU tensor boundary requires two actual compiler sources and profiles', () => {
  assert.deepEqual(c.provides, ['oreslang.tensor.cpu.v1']);
  assert.deepEqual(c.requires, []);
  assert.deepEqual(c.upstreams, [
    'oreslang-common-source','oreslang-source.java','oreslang-source.llvm','oreslang-target-profiles'
  ]);
  assert.ok(!c.upstreams.includes(c.repository));
  assert.match(c.nextGate, /owned contiguous Array<T>/);
  assert.match(c.nextGate, /shape\/stride validation/i);
  assert.match(c.nextGate, /reject aliases, overflow and non-finite values/i);
});

test('metadata is not a substitute for a stepful native tensor guest receipt', () => {
  assert.equal(c.kind, 'PROPOSED_CONTRACT_NOT_IMPLEMENTATION');
  assert.equal(c.hostOracleVerified, false);
  assert.equal(c.guestExecution, 'NOT_EXECUTED');
  assert.deepEqual(c.evidence, []);
  assert.match(c.nextGate, /Compile and execute on Java\/LLVM/);
});
