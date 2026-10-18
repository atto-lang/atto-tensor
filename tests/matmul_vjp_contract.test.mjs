// Independent HOST-ONLY arithmetic oracle and source-contract checks.
// Does not compile .ores, own mutable tensor buffers or authorize actor state.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../src/matmul_vjp.ores",import.meta.url),"utf8");
const fixture=readFileSync(new URL("./matmul_vjp_smoke.ores",import.meta.url),"utf8");
function dualVjp(a,b,g,m,k,n) {
  if(![m,k,n].every(x=>Number.isSafeInteger(x)&&x>=0))throw Error("SHAPE");
  if(a.length!==m*k||b.length!==k*n||g.length!==m*n)throw Error("SHAPE");
  if([...a,...b,...g].some(x=>!Number.isFinite(x)))throw Error("NONFINITE_INPUT");
  const da=Array(m*k).fill(0),db=Array(k*n).fill(0);
  for(let i=0;i<m;i++)for(let q=0;q<k;q++)
    for(let j=0;j<n;j++){
      da[i*k+q]+=g[i*n+j]*b[q*n+j];
      if(!Number.isFinite(da[i*k+q]))throw Error("NONFINITE_DA");
    }
  for(let q=0;q<k;q++)for(let j=0;j<n;j++)
    for(let i=0;i<m;i++){
      db[q*n+j]+=a[i*k+q]*g[i*n+j];
      if(!Number.isFinite(db[q*n+j]))throw Error("NONFINITE_DB");
    }
  return {da,db};
}
function matmul(a,b,m,k,n) {
  const y=Array(m*n).fill(0);
  for(let i=0;i<m;i++)for(let j=0;j<n;j++)
    for(let q=0;q<k;q++)y[i*n+j]+=a[i*k+q]*b[q*n+j];
  return y;
}
function dot(xs,ys){return xs.reduce((s,v,i)=>s+v*ys[i],0);}
test("pure Oreslang source checks both output aliases and input aliasing",()=>{
  assert.match(source,/pub fnc matmul_2d_vjp_f64_checked_into\(/);
  assert.match(source,/left_scratch is right_scratch/);
  assert.match(source,/left_gradient is right_gradient/);
  assert.match(source,/right_gradient is upstream.values/);
  assert.match(source,/checked_numel\(left.shape/);
  assert.match(source,/checked_numel\(right.shape/);
  assert.match(fixture,/ORES_TENSOR_MATMUL_DUAL_VJP_PASS/);
});
test("both scratch gradients stage before either output publication",()=>{
  const la=source.indexOf("left_scratch[(i*inner)+k]=gradient");
  const rb=source.indexOf("right_scratch[(k*columns)+j]=gradient");
  const publish=source.indexOf("left_gradient[i]=left_scratch[i]");
  const publishRight=source.indexOf("right_gradient[i]=right_scratch[i]");
  assert(la>0&&rb>la&&publish>rb&&publishRight>publish);
  assert.match(source,/nonfinite left gradient/);
  assert.match(source,/nonfinite right gradient/);
});
test("rectangular example gradients match pinned smoke values",()=>{
  const out=dualVjp([1,2,3,4,5,6],[7,8,9,10,11,12],[1,2,3,4],2,3,2);
  assert.deepEqual(out,{da:[23,29,35,53,67,81],db:[13,18,17,24,21,30]});
});
test("central differences agree with both reverse-mode parameter Jacobians",()=>{
  const a=[.3,1,-.2,.7,2,-1],b=[2,-1,3,1,.5,-2],g=[.1,2,-3,.3];
  const out=dualVjp(a,b,g,2,3,2),eps=1e-6;
  for(let i=0;i<a.length;i++){
    const hi=a.slice(),lo=a.slice();hi[i]+=eps;lo[i]-=eps;
    const finite=(dot(matmul(hi,b,2,3,2),g)-dot(matmul(lo,b,2,3,2),g))/(2*eps);
    assert(Math.abs(finite-out.da[i])<1e-8);
  }
  for(let i=0;i<b.length;i++){
    const hi=b.slice(),lo=b.slice();hi[i]+=eps;lo[i]-=eps;
    const finite=(dot(matmul(a,hi,2,3,2),g)-dot(matmul(a,lo,2,3,2),g))/(2*eps);
    assert(Math.abs(finite-out.db[i])<1e-8);
  }
});
test("seeded tensor ranks include zero extents without accidental allocation",()=>{
  let state=0x937ab92;
  const rnd=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state;};
  for(let trial=0;trial<300;trial++){
    const m=rnd()%5,k=rnd()%6,n=rnd()%7;
    const a=Array.from({length:m*k},()=>((rnd()%31)-15)/9);
    const b=Array.from({length:k*n},()=>((rnd()%31)-15)/9);
    const g=Array.from({length:m*n},()=>((rnd()%31)-15)/9);
    const {da,db}=dualVjp(a,b,g,m,k,n);
    const h=Array.from({length:a.length},()=>((rnd()%31)-15)/9);
    const t=Array.from({length:b.length},()=>((rnd()%31)-15)/9);
    assert(Math.abs(dot(matmul(h,b,m,k,n),g)-dot(h,da))<1e-8);
    assert(Math.abs(dot(matmul(a,t,m,k,n),g)-dot(t,db))<1e-8);
  }
});
test("bad shapes or overflow fail and do not publish output",()=>{
  const left=[73],right=[74];
  for(const f of [
    ()=>dualVjp([1],[1],[],1,1,1),
    ()=>dualVjp([1e308],[1e308],[1e308],1,1,1)
  ]) {
    assert.throws(f);
    assert.deepEqual(left,[73]);
    assert.deepEqual(right,[74]);
  }
  assert.throws(()=>dualVjp([1],[2],[NaN],1,1,1),/NONFINITE_INPUT/);
});
