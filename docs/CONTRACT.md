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

## New CPU rank-2 matrix adjoint (October 18)

`src/matmul_vjp.ores` provides `matmul_2d_vjp_f64_checked_into(left, right, upstream, left_scratch, right_scratch, left_gradient, right_gradient)`. For `C=A@B`, it computes both adjoints `dA=dC@B^T` and `dB=A^T@dC` without allocating output buffers. Shapes `[M,K]`, `[K,N]`, `[M,N]` are validated using existing `validate_tensor` and `checked_numel`; zero extents are legal. Each result is accumulated with a finite check in its own **caller-owned scratch**. **Both public gradient outputs are written only after BOTH adjoints succeed.** Invalid shapes, nonfinite inputs, arithmetic overflow and output/input/scratch aliases fail closed, leaving both public outputs unchanged; scratch is expendable.

This adds actual Oreslang source functions, not independent ndarray host functionality. It is a fundamental CPU kernel for later multi-layer tensor backprop. The existing `List<float>` backing remains mutable and does **not** prove unique ownership, zero-copy I/O, actor-safe sendability, reliable storage lease identities, or parallel execution.

`tests/matmul_vjp_smoke.ores` is an executable guest acceptance target for rectangular numeric values, invalid shapes, illegal aliases, arithmetic overflow and zero-inner-dimension matrices. `tests/matmul_vjp_contract.test.mjs` is a separate **host-only** 6-case reference, including seeded adjoint identities and independent central differences on both matrices. Isolated V8 checks passed 6/6 on source Git blob `0fb9488062923660c163e44f9a45f69db9dacdef` and test blob `52ad1831d631ded4c4572b1fb48613465bf7f248`; this is **not** a successful guest compiler result.

The existing source-org Actions failures provide no positive guest evidence. An independently authored source-free dual-matmul mathematical oracle also ran as part of the funded [31/31 host-only workflow](https://github.com/sagitta-stack/sagitta-infra/actions/runs/39154221794), which copied **no** private Oreslang source. Java/Graal, LLVM and Wasm backend execution each remain separate mandatory release gates. Do not mirror private Git objects outside authorized source boundaries while Oreslang security incidents #22 and #77 remain open.

The mathematically computed gradients are **not automatically state mutations**. Model stage actors must retain all weight handles, optimizer slots and gradient receipts under exactly one `state {}`; access requires `with` effect declarations and scoped `using state.read` or `using state.write(critical ...)` / sealed `handle_state`. Compute matrix adjoints outside state RW loans on verified immutable pinned snapshots; publish full optimizer candidates inside a short version-fenced critical loan.


### Audit hardening: asymmetric overflow and pinned guest admission

Added a separate **right-only overflow** guest fixture where `A=[1e308]`, `B=[0]`, and `dC=[1e308]`. The `dA` calculation is finite zero, while `dB` overflows **after the entire left scratch has been prepared**; neither externally visible gradient output may be updated. This closes a test coverage gap: a prior huge-all-operands fixture could fail during the left adjoint and never exercise the two-output publication barrier.

The independent host oracle now has **8/8** passing isolated checks (exact test blob `7775ec2eeba69acdcc6a7ee5e3da77b161ee73c1`; actual Oreslang source blob `0fb9488062923660c163e44f9a45f69db9dacdef`; guest fixture `5aced20ccd0bf8dfd38beb35b13c94decc7fd714`). These are mathematical + lexical source checks, **not native execution**.

New `scripts/verify-matmul-vjp-guest.sh` requires an independently clean Git checkout at `ORESLANG_TENSOR_EXPECTED_SHA`, a real absolute executable with a matching `ORESLANG_RUNNER_SHA256`, tracked imported source files, and exact guest stdout `ORES_TENSOR_MATMUL_DUAL_VJP_PASS`. Three isolated host-only gate checks passed. The executable hash alone does not bind dynamically loaded libraries/JARs, JDK/native runtime artifacts or flags; native build receipts must cover those separately.

The independent **source-free** funded [Actions run #39155779350](https://github.com/sagitta-stack/sagitta-infra/actions/runs/39155779350) passed 40/40 aggregate host tests at exact carrier SHA `aa026e01b9479694906817b570f250edb09bd550`, including a source-free asymmetric overflow oracle. Logs explicitly confirm `PRIVATE_ORESLANG_SOURCE=NOT_TRANSFERRED`, `ORESLANG_GUEST_EXECUTION=NOT_EXECUTED`, and `EXACT_SOURCE_TREE=NOT_MIRRORED`. Source-org CI continues failing before usable steps, and security incidents #22/#77 preclude new cross-org private-source mirrors absent independent authorization.
