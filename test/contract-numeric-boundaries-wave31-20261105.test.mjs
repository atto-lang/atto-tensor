import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const contract = JSON.parse(readFileSync(new URL('../integration/contract.v1.json', import.meta.url), 'utf8'));

test('tensor contract requires three separately pinned scalar fixtures', () => {
  assert.deepEqual(contract.fixtureIds, ['tensor/add/v1', 'tensor/matmul/v1', 'tensor/shape/v1']);
  for (const id of contract.fixtureIds) {
    assert.match(id, /^tensor\/[a-z][a-z0-9-]*\/v[1-9][0-9]*$/);
    assert.ok(!id.split('/').includes('..'));
  }
});

test('tensor owner and backend toolchain dependencies cannot be silently elided', () => {
  assert.deepEqual(contract.provides, ['oreslang.tensor.cpu.v1']);
  assert.deepEqual(contract.requires, []);
  assert.deepEqual(contract.upstreams, [
    'oreslang-common-source', 'oreslang-source.java',
    'oreslang-source.llvm', 'oreslang-target-profiles',
  ]);
});

test('tensor layout host metadata is not guest execution or alias-safety certification', () => {
  assert.equal(contract.schemaVersion, 1);
  assert.equal(contract.kind, 'PROPOSED_CONTRACT_NOT_IMPLEMENTATION');
  assert.equal(contract.hostOracleVerified, false);
  assert.equal(contract.guestExecution, 'NOT_EXECUTED');
  assert.deepEqual(contract.evidence, []);
  assert.match(contract.nextGate, /owned contiguous Array<T>/);
  assert.match(contract.nextGate, /reject aliases, overflow and non-finite values/i);
});
