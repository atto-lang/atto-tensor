// Host-only source/provenance gate for a future real Oreslang compiler run.
// It does not verify the compiler/JDK/native library transitive dependency set.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../scripts/verify-matmul-vjp-guest.sh",import.meta.url),"utf8");
test("requires exact tensor head and executable SHA-256",()=>{
  assert.match(source,/ORESLANG_TENSOR_EXPECTED_SHA/);
  assert.match(source,/ORESLANG_RUNNER_SHA256/);
  assert.match(source,/git -C "\$root" rev-parse HEAD/);
  assert.match(source,/if \[\[ "\$actual" != "\$ORESLANG_TENSOR_EXPECTED_SHA" \]\]/);
  assert.match(source,/runner_hash.*ORESLANG_RUNNER_SHA256/);
});
test("rejects working tree changes, ignored missing fixtures and fake runners",()=>{
  assert.match(source,/status --porcelain --untracked-files=all/);
  assert.match(source,/git -C "\$root" ls-files --error-unmatch/);
  assert.match(source,/src\/matmul_vjp\.ores/);
  assert.match(source,/tests\/matmul_vjp_smoke\.ores/);
  assert.match(source,/\! -x "\$runner"/);
});
test("does not accept nonempty error output or mismatched guest results",()=>{
  assert.match(source,/ORES_TENSOR_MATMUL_DUAL_VJP_PASS/);
  assert.match(source,/diff -u "\$tmp\/expected" "\$tmp\/stdout"/);
  assert.match(source,/ORESLANG_TENSOR_MATMUL_GUEST=FAILED/);
  assert.match(source,/ORESLANG_TENSOR_MATMUL_GUEST=OUTPUT_MISMATCH/);
  assert.match(source,/ORESLANG_TENSOR_MATMUL_GUEST=PASS/);
});
