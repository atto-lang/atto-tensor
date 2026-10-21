# Oreslang Tensor

Native Oreslang tensor semantics and CPU reference kernels. The first tranche uses plain `.ores` source and `List<T>` backing storage; this is not yet an ownership-safe contiguous `Array<T>` implementation or an accelerated backend.

## Scope (P0)
- Checked shape/product and contiguous tensor descriptors.
- Scalar reference kernels with caller-supplied outputs.
- No Python/Java numerical implementation in public tensor algorithms.
- Unsafe aliases, unsupported devices, overflow and shape mismatches must fail closed.

See `docs/CONTRACT.md` and `src/tensor.ores` after bootstrap PR.

## Engineering policy: no Python

Python is prohibited in this repository and throughout the `ores-truffle-oreslang` organization for scripts, tests, CI helpers, generators, analyzers, automation, and host tooling. Existing Python functionality must be migrated to **Rust** without weakening behavior, fixtures, or test coverage. The tensor implementation remains Oreslang; this rule specifically replaces Python tooling/reference code with Rust.
