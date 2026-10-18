// HOST-ONLY independent numerical oracle for row minibatching and gather VJP.
// This runs Node 22, NOT an Oreslang compiler, actor or GPU backend.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../src/row_batches.ores", import.meta.url),"utf8");
const guest = readFileSync(new URL("./row_batches_smoke.ores", import.meta.url),"utf8");
const finite = n => Number.isFinite(n);
function shape(rows, cols, data) {
  if (!Number.isSafeInteger(rows)||!Number.isSafeInteger(cols)||rows<0||cols<0)
    throw Error("BAD_SHAPE");
  if (!Array.isArray(data)||rows*cols!==data.length||!data.every(finite))
    throw Error("BAD_TENSOR");
  return {rows,cols,data:Object.freeze([...data])};
}
function gather(input,indices) {
  if (!Array.isArray(indices)||indices.some(i=>!Number.isSafeInteger(i)||i<0||i>=input.rows))
    throw Error("BAD_INDEX");
  const data=[];
  for(const row of indices)for(let c=0;c<input.cols;c++)data.push(input.data[row*input.cols+c]);
  return shape(indices.length,input.cols,data);
}
function slice(input,start,count) {
  if(!Number.isSafeInteger(start)||!Number.isSafeInteger(count)||
     start<0||count<0||start>input.rows||count>input.rows-start)throw Error("BAD_SLICE");
  return shape(count,input.cols,input.data.slice(start*input.cols,(start+count)*input.cols));
}
function vjp(input,indices,upstream) {
  if(upstream.rows!==indices.length||upstream.cols!==input.cols)throw Error("BAD_UPSTREAM");
  // Validate ALL indices before starting accumulation.
  for(const r of indices)if(!Number.isSafeInteger(r)||r<0||r>=input.rows)throw Error("BAD_INDEX");
  const out=new Array(input.rows*input.cols).fill(0);
  for(let j=0;j<indices.length;j++)for(let c=0;c<input.cols;c++){
    const idx=indices[j]*input.cols+c;
    const value=out[idx]+upstream.data[j*input.cols+c];
    if(!finite(value))throw Error("NONFINITE_GRADIENT");
    out[idx]=value;
  }
  return shape(input.rows,input.cols,out);
}
function approx(a,b,tol=1e-7) { assert(Math.abs(a-b)<=tol, `${a} not ≈ ${b}`); }
let seed=0x9e3779b9;
function rand() {seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return(seed>>>0)/4294967296;}
function randint(n) {return Math.floor(rand()*n);}
function dot(a,b) {assert.equal(a.length,b.length);return a.reduce((s,x,i)=>s+x*b[i],0);}

test("source API, manual output preflight, disjoint scratch and real guest fixture are present",()=>{
  for(const fn of ["gather_rows_2d_f64_into","slice_rows_2d_f64_into",
                   "gather_rows_vjp_f64_checked_into"]) {
    assert.match(source,new RegExp("pub fnc "+fn+"\\("));
    assert(guest.includes(fn+"("));
  }
  assert.match(source,/output is source\.values/);
  assert.match(source,/scratch is output/);
  assert.match(source,/checked_numel/);
  assert.match(guest,/ORES_TENSOR_ROW_BATCH_VJP_PASS/);
});

test("gather preserves index order, duplicates, and independent immutable input",()=>{
  const matrix=shape(3,2,[1,2,3,4,5,6]);
  const selected=gather(matrix,[2,0,2]);
  assert.deepEqual(selected.data,[5,6,1,2,5,6]);
  assert.deepEqual(matrix.data,[1,2,3,4,5,6]);
  assert.deepEqual(slice(matrix,1,2).data,[3,4,5,6]);
  assert.deepEqual(slice(matrix,3,0).data,[]);
});

test("scatter-add VJP accumulates each duplicate source-row contribution",()=>{
  const x=shape(3,2,[1,2,3,4,5,6]);
  const dx=vjp(x,[2,0,2],shape(3,2,[1,2,3,4,5,6]));
  assert.deepEqual(dx.data,[3,4,0,0,6,8]);
});

test("zero row and zero column shapes do not divide by zero or allocate",()=>{
  assert.deepEqual(gather(shape(0,3,[]),[]).data,[]);
  assert.deepEqual(vjp(shape(0,3,[]),[],shape(0,3,[])).data,[]);
  assert.deepEqual(gather(shape(4,0,[]),[3,0,3]).data,[]);
  assert.deepEqual(vjp(shape(4,0,[]),[3,0,3],shape(3,0,[])).data,[]);
});

test("bad rank/length/index/slice rejects rather than truncating or padding",()=>{
  assert.throws(()=>shape(2,2,[1,2,3]),/BAD_TENSOR/);
  assert.throws(()=>shape(-1,2,[]),/BAD_SHAPE/);
  assert.throws(()=>shape(1,1,[Infinity]),/BAD_TENSOR/);
  const x=shape(3,2,[1,2,3,4,5,6]);
  for(const ids of [[-1],[3],[1,5],[0.5],[NaN]])assert.throws(()=>gather(x,ids),/BAD_INDEX/);
  for(const [a,b] of [[-1,1],[3,1],[2,2],[4,0],[0,-1],[0,1.5]])
    assert.throws(()=>slice(x,a,b),/BAD_SLICE/);
  assert.throws(()=>vjp(x,[0,1],shape(1,2,[1,2])),/BAD_UPSTREAM/);
  assert.throws(()=>vjp(x,[3],shape(1,2,[1,2])),/BAD_INDEX/);
});

test("gather VJP linear-adjoint identity over seeded random shapes and indices",()=>{
  for(let k=0;k<350;k++){
    const rows=randint(7),cols=randint(6),count=rows===0?0:randint(9);
    const x=shape(rows,cols,Array.from({length:rows*cols},()=>rand()*8-4));
    const indices=Array.from({length:count},()=>randint(rows));
    const dy=shape(count,cols,Array.from({length:count*cols},()=>rand()*8-4));
    const g=gather(x,indices),dx=vjp(x,indices,dy);
    approx(dot(g.data,dy.data),dot(x.data,dx.data),1e-9);
    for(let j=0;j<count;j++)for(let c=0;c<cols;c++)
      assert.equal(g.data[j*cols+c],x.data[indices[j]*cols+c]);
  }
});

test("numerical finite-difference VJP for repeated row sampling",()=>{
  const x=shape(3,2,[1,2,-3,4,5,-6]);
  const indices=[2,0,2];
  const upstream=shape(3,2,[2,-1,4,3,-2,5]);
  const actual=vjp(x,indices,upstream);
  const eps=1e-5;
  for(let i=0;i<x.data.length;i++){
    const plus=x.data.slice(),minus=x.data.slice();
    plus[i]+=eps;minus[i]-=eps;
    const fPlus=dot(gather(shape(x.rows,x.cols,plus),indices).data,upstream.data);
    const fMinus=dot(gather(shape(x.rows,x.cols,minus),indices).data,upstream.data);
    approx((fPlus-fMinus)/(2*eps),actual.data[i],2e-9);
  }
});

test("nonfinite scatter accumulation must fail instead of publishing Infinity",()=>{
  const x=shape(1,1,[0]);
  const upstream=shape(2,1,[1e308,1e308]);
  assert.throws(()=>vjp(x,[0,0],upstream),/NONFINITE_GRADIENT/);
});
