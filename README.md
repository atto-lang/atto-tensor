# Oreslang Tensor

Native Oreslang tensor semantics and CPU reference kernels. The first tranche uses plain `.ores` source and `List<T>` backing storage; this is not yet an ownership-safe contiguous `Array<T>` implementation or an accelerated backend.

## Scope (P0)
- Checked shape/product and contiguous tensor descriptors.
- Scalar reference kernels with caller-supplied outputs.
- No Python/Java numerical implementation in public tensor algorithms.
- Unsafe aliases, unsupported devices, overflow and shape mismatches must fail closed.

See `docs/CONTRACT.md` and `src/tensor.ores` after bootstrap PR.
