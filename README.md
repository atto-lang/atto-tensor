# atto-tensor

Tensor data structures and numerical computation primitives for AttoLang.

## Naming migration and language authority

This repository is maintained at [atto-lang/atto-tensor](https://github.com/atto-lang/atto-tensor); use `https://github.com/atto-lang/atto-tensor.git` for Git remotes and canonical `.atto` source examples. The normative language contract is [atto-spec](https://github.com/atto-lang/atto-spec), with executable guest evidence tracked by [atto-platform-conformance](https://github.com/atto-lang/atto-platform-conformance). Follow current ownership, borrowing, effects, actor, and syntax rules there; do not infer implemented features from a README or unexecuted CI.

## Scope

Tensor shapes, data layouts, numerical operations, and ownership-aware memory contracts.

## API and portability principles

- Specify numerical/semantic behavior before assuming support on every backend.
- Document input types, error behavior, shape and memory ownership requirements as applicable.
- Distinguish CPU execution, SIMD, GPU offload, and other target-specific capabilities.

## Verification

Use checked-in tests, documented tolerances, and exact-commit runtime evidence. Host-only or compilation tests are not equivalent to executed AttoLang guest tests. Record backend gaps explicitly.

## Development status

This describes the repository's intended purpose, not a verified list of implemented features. Refer to checked-in code and CI for current functionality and validated commands.

## Related

- [AttoLang](https://github.com/atto-lang)
- [ML conformance](https://github.com/atto-lang/atto-ml-conformance)
- [Language specification](https://github.com/atto-lang/atto-spec)

## Repository ownership and contribution routing

**Belongs here:** Tensor storage, shapes/strides, layout transformations, broadcasting, numerical kernels and ownership-safe memory views.

**Belongs elsewhere:** Gradient graphs in atto-autograd; model APIs in atto-ml; data analysis pipelines in atto-data-science.

**Security and portability boundary:** Reject shape/stride overflow, invalid views, aliasing and use-after-free; bounded allocations and target-specific overflow/precision tests.

AttoLang is the canonical project name; new source, examples and documentation use `.atto` and the `atto-lang` repository organization. This README defines scope, not feature readiness.

**Acceptance:** name exact source SHA/tree, runtime/toolchain, positive and negative checks and execution receipt before claiming support. Host-only tests and empty CI steps do not prove guest execution.
