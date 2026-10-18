import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const c=JSON.parse(readFileSync(new URL("../integration/contract.v1.json",import.meta.url),"utf8"));
test("contract identifies exactly one source organization/repository",()=>{
 assert.equal(c.schemaVersion,1);assert.equal(c.organization,"ores-truffle-oreslang");
 assert.equal(c.repository,process.env.npm_package_name.replace(/^@oreslang\//,""));
 assert.match(c.mainSnapshotSha,/^[a-f0-9]{40}$/);
});
test("published interface IDs and upstream edges are unique and canonical",()=>{
 for(const property of ["provides","requires","upstreams","fixtureIds"]){
  assert.ok(Array.isArray(c[property]));
  assert.deepEqual(c[property],[...new Set(c[property])].sort());
 }
 assert.ok(c.provides.length>0);
 assert.ok(c.fixtureIds.length>0);
 assert.ok(!c.upstreams.includes(c.repository));
 for(const req of c.requires) assert.ok(!c.provides.includes(req));
});
test("no host-oracle pass is elevated to Oreslang runtime proof",()=>{
 assert.equal(c.kind,"PROPOSED_CONTRACT_NOT_IMPLEMENTATION");
 assert.equal(c.hostOracleVerified,false);
 assert.equal(c.guestExecution,"NOT_EXECUTED");
 assert.deepEqual(c.evidence,[]);
});
test("next acceptance step must require executable semantics",()=>{
 assert.ok(c.nextGate.length>60);
 assert.match(c.nextGate,/(implement|compile|execute)/i);
});
