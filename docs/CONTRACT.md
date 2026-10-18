# Tensor P0 contract (proposal, not yet proven compiler parity)

## Current supported reference operations
- CPU row-major dense `TensorF64(shape: List<int>, values: List<float>)`, rank 0+.
- `checked_numel` checks negative dimensions, zero extents and exact storage extent without unchecked shape-product multiplication.
- Same-shape `add_f64_into`, elementwise `relu_f64_into`, `sum_f64`, and rank-2 `matmul_2d_f64_into`.
- All computational functions are in `.ores` and return `Result`; caller supplies output storage, so the reference kernels do not allocate result arrays.

## Honest limitations
- `List<float>` is a current source-language stand-in. It does not assert verified f64 bits, contiguous buffers, or allocator/GC guarantees across all backends.
- Class fields expose List references, and aliases can mutate them. The caller MUST NOT alias output buffers with inputs or mutate descriptors/data during operations. This is a **temporary precondition**, not a demonstrated borrow-checker safety guarantee.
- `checked_numel` intentionally uses bounded repeated addition rather than potentially overflowing multiplication. This is a correctness reference, not a performance implementation.
- No generic `Tensor<T>`, dtype dispatch, striding, broadcasting, slicing, GPU, SIMD, autograd, or cross-backend execution claims yet.
- IEEE exceptional values/overflow need separate normative tests and backend admission before release.

## Next gate
1. Implement compiler-recognized unique/frozen contiguous `Array<T>` and checked projected views, including take/share/borrow and alias rejection.
2. Enforce whole-buffer write loans and validate complete output before mutation; track cancellation and lifetime for device buffers.
3. Bring `oreslang-data-science` numeric APIs into a shared layout contract instead of reimplementing BLAS.
4. Execute `tests/smoke.ores` with an exact pinned real Java/Graal compiler. Then add LLVM and Wasm only when they compile the identical source.
