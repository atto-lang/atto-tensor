# Tensor P0 contract (proposal, not yet proven compiler parity)

## Current supported reference operations
- CPU row-major dense `TensorF64(shape: List<int>, values: List<float>)`, rank 0+.
- `checked_numel` checks negative dimensions, zero extents and exact storage extent without unchecked shape-product multiplication.
- Same-shape `add_f64_into`, elementwise `relu_f64_into`, `sum_f64`, and rank-2 `matmul_2d_f64_into`.
- All computational functions are in `.ores` and return `Result`; caller supplies output storage, so the reference kernels do not allocate result arrays.

## Honest limitations
- `List<float>` is a current source-language stand-in. It does not assert verified f64 bits, contiguous buffers, or allocator/GC guarantees across all backends.
- Class fields expose List references, and aliases can mutate them. The caller MUST NOT alias output buffers with inputs or mutate descriptors/data during operations. This is a **temporary precondition**, not a demonstrated borrow-checker safety guarantee.
- `checked_numel` is now O(rank): division checks the storage bound before multiplication, so huge extents cannot cause dimension-sized loops or product overflow.
- P0 rejects nonfinite input values and preflights all arithmetic results before writing caller outputs; the matmul reference deliberately computes twice for this guarantee. Concurrent mutation of aliased List storage remains unsafe until the borrow checker proves exclusivity.
- `src/broadcast.ores` supports validated right-aligned dense addition and multiplication, including scalar expansion and zero extents. `src/gradients.ores` provides checked scratch-backed sum-to-shape VJP reduction.
- No generic `Tensor<T>`, dtype dispatch, strided views, slicing, GPU, SIMD, full tensor autograd, or cross-backend execution claims yet.
- IEEE exceptional values/overflow need separate normative tests and backend admission before release.

## Next gate
1. Implement compiler-recognized unique/frozen contiguous `Array<T>` and checked projected views, including take/share/borrow and alias rejection.
2. Enforce whole-buffer write loans and validate complete output before mutation; track cancellation and lifetime for device buffers.
3. Bring `oreslang-data-science` numeric APIs into a shared layout contract instead of reimplementing BLAS.
4. Execute `tests/smoke.ores` with an exact pinned real Java/Graal compiler. Then add LLVM and Wasm only when they compile the identical source.

## P1 differentiable minibatch row operations (stacked draft, October 18, 2026)

`src/row_batches.ores` adds three pure-Oreslang, CPU-only reference kernels:

| API | Data/ML responsibility | Differentiable behavior |
| --- | --- | --- |
| `gather_rows_2d_f64_into(source, indices, output)` | DataLoader-style selected and reordered minibatches; repeated indices allowed | VJP via `gather_rows_vjp_f64_checked_into` |
| `slice_rows_2d_f64_into(source, start, count, output)` | Contiguous, allocation-free batch copying | Equivalent gradient is scatter to contiguous source range (not yet exported as a separate VJP) |
| `gather_rows_vjp_f64_checked_into(upstream, indices, source_shape, scratch, output)` | Reverse-mode gradient of row gather | **Sum** repeated-index contributions deterministically in index order |

These operations keep training/data math out of individual actors. They can be executed between actor messages while the actor's sole `state { ... }` holds only immutable parameter-snapshot handles, activation/gradient ledgers and receipts. `with` declares effects; `using state.read` and `using state.write(critical ...)` acquire scoped state capabilities. No state loan remains live during a tensor kernel, backpressure wait, or message dispatch. This draft does not add another actor state field or a second lock mechanism.

### Resource, shape, and failure guarantees

- Source and upstream tensors use existing `validate_tensor` (finite values and checked row-major shapes).
- Original and target extents are validated via `checked_numel`; bounds use `count > rows - start` rather than potentially overflowing `start + count`.
- Invalid/negative indices, rank mismatch, source/output alias and undersized/oversized buffers return `Err`.
- Gather and slice preflight all invalid inputs **before** writing even one output element.
- VJP permits repeated sampled rows; accumulates in caller-owned scratch, checks every addition is finite and publishes public output only on successful completion. Scratch may be modified on failed accumulation.
- Empty dataset, zero-feature tensor, and empty selections are legal when dimensions and storage agree.
- No implicit GPU, allocation, I/O, device transfer, parallel execution, shared mutable alias, or cross-actor pointer export.
- The current `List<float>` representation does NOT prove unique buffer loans, safe concurrent mutation, an actual f64 ABI or `no_std` freestanding compatibility. Those require compiler/runtime evidence.

### Test evidence and promotion gate

- `tests/row_batches_smoke.ores` is the guest acceptance fixture. `scripts/test.sh` now invokes both existing tensor smoke and this fixture **only when `ORESLANG_BIN` is an explicitly supplied real Oreslang executable**, otherwise it exits UNCONFIGURED (78).
- `tests/row_batches_contract.test.mjs` executes 8 host-side tests with an independent Node-compatible numerical oracle, seeded randomized adjoint identities and central finite-difference gradients. It is `HOST_ONLY`; test source presence cannot be called guest compilation.
- Promotion requires pinned exact-tree Java/Graal guest execution, comparison of golden marker/output, negative alias/shape tests, numeric gradient evidence from actual Oreslang, and native LLVM compilation/sanitizers when the backend supports the same source. CI without a runner is NOT evidence of success.
- Combine these row operations with autograd's tensor graph VJP, then train on small deterministic multi-batch datasets, and only after that introduce pipeline-scheduled execution with bounded mailbox/byte credits, cancellation and version fencing.


## P1 checked two-sided matmul VJP (stacked draft)

`src/matmul_vjp.ores` adds pure-Oreslang `matmul_both_vjp_f64_checked_into(X[M,K], W[K,N], dY[M,N], left_scratch, right_scratch, dX_out, dW_out)`, computing `dX = dY @ transpose(W)` and `dW = transpose(X) @ dY` for rank-2 CPU tensors, including valid empty shapes.

The kernel checks all tensor descriptors, shape products, finite source values, expected rank/inner dimensions, output capacities and explicit `List<float>` identity-disjointness before arithmetic. Both gradients are calculated in independent caller-owned scratch buffers with finite product and reduction checks **before either public output is written**. If right-side accumulation overflows after left-side scratch succeeds, both public outputs remain unchanged; scratch may change. This provides logical input/arithmetic-error preservation, NOT crash-atomic dual-buffer publication under process termination or unsafe concurrent memory writes.

The API does not provide a general transpose view, matmul engine, GPU kernel, optimizer update, frozen `TensorF64` storage or actor-state lock. A production actor must pin immutable parameter-version leases with the sole `state {}` and short `using`/`handle_state` RW loans, and run the entire matmul/VJP operation **outside** those loans. No actor-state loan may be sent in a message or kept across suspension.

Native `tests/matmul_two_sided_vjp_smoke.ores` is added to `scripts/test.sh` whenever a real `ORESLANG_BIN` is set. Host-only `tests/matmul_two_sided_vjp_oracle.test.mjs` checks rectangular gradients against central finite differences, seeded bilinear adjoints, zero extents, mismatched shapes and overflow preserving output. Real Java/LLVM guest execution and backend sanitizer evidence remain required and NOT_VERIFIED. Cross-org private compiler tree mirroring is paused under security issues #22 and #77.
