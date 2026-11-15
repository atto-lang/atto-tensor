# atto-tensor

Tensor data structures and numerical computation primitives for AttoLang.

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

AttoLang is the public name; preserve existing Oreslang/.ores compatibility and use canonical .atto for new portable library source. This README defines scope, not feature readiness.

**Acceptance:** name exact source SHA/tree, runtime/toolchain, positive and negative checks and execution receipt before claiming support. Host-only tests and empty CI steps do not prove guest execution.

## Host reference: shape v0 (not an Atto runtime implementation)

`reference/shape_rules.py` defines bounded shape validation, scalar/zero-axis element counts, row-major strides, trailing-axis broadcast compatibility and broadcast index mapping. It rejects booleans as dimensions, negative dimensions, rank over 64 and signed-64-bit element-count overflow. This establishes deterministic expected behavior for future Atto tensor backends; no CPU/GPU/SIMD Atto guest execution is claimed.

Run `python3 -m unittest discover -s tests -v` from the repository root to check the Python reference examples. Before porting, align error representation and resource bounds across compilers.
