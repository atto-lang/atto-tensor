// Host-only independent 2D matmul differentiation reference, not the .ores runtime.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../src/matmul_vjp.ores",import.meta.url),"utf8");
function validate(A,B,G,m,k,n) {
  if(![m,k,n].every(x=>Number.isSafeInteger(x)&&x>=0))throw Error("SHAPE");
  if(A.length!==m*k||B.length!==k*n||G.length!==m*n)throw Error("SHAPE");
  for(const values of [A,B,G])if(values.some(x=>!Number.isFinite(x)))throw Error("NONFINITE");
}
function product(A,B,m,k,n) {
  if(A.length!==m*k||B.length!==k*n)throw Error("SHAPE");
  const out=Array(m*n).fill(0);
  for(let i=0;i<m;i++)for(let j=0;j<n;j++)for(let p=0;p<k;p++)
    out[i*n+j]+=A[i*k+p]*B[p*n+j];
  return out;
}
function bothVjp(A,B,G,m,k,n,dxOld,dwOld) {
  validate(A,B,G,m,k,n);
  const dx=Array(m*k).fill(0),dw=Array(k*n).fill(0);
  if(dxOld.length!==dx.length||dwOld.length!==dw.length)throw Error("CAPACITY");
  for(let i=0;i<m;i++)for(let p=0;p<k;p++)for(let j=0;j<n;j++){
    const term=G[i*n+j]*B[p*n+j],v=dx[i*k+p]+term;
    if(!Number.isFinite(term)||!Number.isFinite(v))throw Error("OVERFLOW_LEFT");
    dx[i*k+p]=v;
  }
  for(let p=0;p<k;p++)for(let j=0;j<n;j++)for(let i=0;i<m;i++){
    const term=A[i*k+p]*G[i*n+j],v=dw[p*n+j]+term;
    if(!Number.isFinite(term)||!Number.isFinite(v))throw Error("OVERFLOW_RIGHT");
    dw[p*n+j]=v;
  }
  // A validated result is published in one logical transaction. A real
  // runtime must separately exclude process termination and unsafe mutation.
  return {dx,dw};
}
function objective(A,B,G,m,k,n) {
  const Y=product(A,B,m,k,n);
  return Y.reduce((sum,v,i)=>sum+v*G[i],0);
}
test("Oreslang source stages both outputs before any public mutation",()=>{
  assert.match(source,/pub fnc matmul_both_vjp_f64_checked_into\(/);
  assert.match(source,/output_left is output_right/);
  assert.match(source,/scratch_right\[\(k \* cols\) \+ col\] = value/);
  const right=source.indexOf("// Compute dW.");
  const publication=source.indexOf("// The public outputs");
  assert(right>0&&publication>right);
  assert.doesNotMatch(source,/\bself\.state\b/);
});
test("2x2 hand-derived left and right gradients",()=>{
  const {dx,dw}=bothVjp([1,2,3,4],[5,6,7,8],[1,2,3,4],2,2,2,[0,0,0,0],[0,0,0,0]);
  assert.deepEqual(dx,[17,23,39,53]);
  assert.deepEqual(dw,[10,14,14,20]);
});
test("asymmetric shape exercises both transposed derivatives",()=>{
  const X=[1,2,3,4,5,6],W=[7,8,9,10,11,12],G=[1,2,3,4];
  const r=bothVjp(X,W,G,2,3,2,Array(6).fill(0),Array(6).fill(0));
  assert.deepEqual(r.dx,[23,29,35,53,67,81]);
  assert.deepEqual(r.dw,[13,18,17,24,21,30]);
});
test("upstream shape mismatch, nonfinite values and capacity mismatches reject",()=>{
  assert.throws(()=>bothVjp([1],[1],[],1,1,1,[0],[0]),/SHAPE/);
  assert.throws(()=>bothVjp([NaN],[1],[1],1,1,1,[0],[0]),/NONFINITE/);
  assert.throws(()=>bothVjp([1],[1],[1],1,1,1,[],[0]),/CAPACITY/);
});
test("overflow in second gradient branch leaves published outputs unchanged",()=>{
  const oldX=[17],oldW=[23];
  assert.throws(()=>bothVjp([1e308],[1],[1e308],1,1,1,oldX,oldW),/OVERFLOW_RIGHT/);
  assert.deepEqual(oldX,[17]);assert.deepEqual(oldW,[23]);
});
test("zero dimensions emit zero or empty gradients as algebra requires",()=>{
  const a=bothVjp([], [1,2,3,4,5,6], [],0,2,3,[],Array(6).fill(99));
  assert.deepEqual(a.dx,[]);
  assert.deepEqual(a.dw,Array(6).fill(0));
  const b=bothVjp([],[],[],3,0,2,[],[]);
  assert.deepEqual(b.dx,[]); assert.deepEqual(b.dw,[]);
});
test("finite differences agree for rectangular matrices and both operands",()=>{
  const m=2,k=3,n=2,A=[1,2,3,4,5,6],B=[7,8,9,10,11,12],G=[1,2,3,4],h=1e-5;
  const r=bothVjp(A,B,G,m,k,n,Array(m*k).fill(0),Array(k*n).fill(0));
  for(let q=0;q<A.length;q++){
    const hi=A.slice(),lo=A.slice();hi[q]+=h;lo[q]-=h;
    const num=(objective(hi,B,G,m,k,n)-objective(lo,B,G,m,k,n))/(2*h);
    assert(Math.abs(num-r.dx[q])<1e-7);
  }
  for(let q=0;q<B.length;q++){
    const hi=B.slice(),lo=B.slice();hi[q]+=h;lo[q]-=h;
    const num=(objective(A,hi,G,m,k,n)-objective(A,lo,G,m,k,n))/(2*h);
    assert(Math.abs(num-r.dw[q])<1e-7);
  }
});
test("seeded rank-2 adjoints pass across empty and repeated shapes",()=>{
  let s=1901743;const next=()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s;};
  for(let trial=0;trial<320;trial++){
    const m=next()%6,k=next()%5,n=next()%7;
    const A=Array.from({length:m*k},()=>((next()%21)-10)/7);
    const B=Array.from({length:k*n},()=>((next()%21)-10)/7);
    const G=Array.from({length:m*n},()=>((next()%21)-10)/7);
    const r=bothVjp(A,B,G,m,k,n,Array(m*k).fill(0),Array(k*n).fill(0));
    const U=Array.from({length:m*k},()=>((next()%21)-10)/7);
    const V=Array.from({length:k*n},()=>((next()%21)-10)/7);
    const left=objective(U,B,G,m,k,n);
    const right=U.reduce((sum,x,i)=>sum+x*r.dx[i],0);
    const leftOther=objective(A,V,G,m,k,n);
    const rightOther=V.reduce((sum,x,i)=>sum+x*r.dw[i],0);
    assert(Math.abs(left-right)<1e-8);
    assert(Math.abs(leftOther-rightOther)<1e-8);
  }
});
