> **AI agent instructions (organization-wide):** Before working on code, issues, or pull requests, read and follow the [organization `AGENTS.md`](https://github.com/atto-lang/.github/blob/main/AGENTS.md), including its **merge-first stealth-development policy**. Honor any repository-local `AGENTS.md`, explicit holds, and GitHub-enforced rules.

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

## CPU tensor kernels

Import the `TensorF64` class and `TensorOps` module from `src/tensor.ores` for
shape validation, elementwise add, ReLU, sum, and matrix multiplication.
`Broadcast`, `TensorGradients`, `RowBatches`, `MatmulVjp`, and `MatmulBothVjp`
expose broadcasting, gradient reduction, row gather and slice, and matrix
gradients. Pass inputs with `rt borrow` and caller-owned scratch and output
storage with `rt mut borrow`. Tensors use finite f64 values in row-major order.

`Softmax.softmax_last_axis_f64_into` normalizes the last axis of any
positive-rank tensor, with a maximum shift before exponentiation. Its checked
VJP uses the same input shape and recomputes the probabilities from the logits.
[Softmax numerical analysis](https://arxiv.org/abs/1909.03469)

`Attention.scaled_dot_product_f64_into` accepts queries shaped
`[..., queries, depth]`, keys shaped `[..., keys, depth]`, and values shaped
`[..., keys, channels]`. The leading batch and head dimensions must match.
It writes `[..., queries, channels]` and accepts a causal flag that limits
each query to keys at or before its position within that batch. Score scratch
has shape `[..., queries, keys]`. The checked VJP publishes the three input
gradients together and needs scratch for scores, score gradients, and the
concatenated query, key, and value gradients. The forward and reverse kernels
use caller-owned, disjoint row-major buffers.
[Scaled dot-product attention](https://arxiv.org/abs/1706.03762)

## Packed AoS and SoA

`src/layout.atto` exposes explicit conversions for packed homogeneous f64
records with logical shape `[rows, fields]`. AoS stores each row together.
SoA stores each field together. Callers provide disjoint storage through
read and exclusive mutable borrows. The conversion checks both extents and
storage lengths before writing, and the inverse permutation is its VJP.

The layout guest fixture checks the physical order, round trip, adjoint
identity, and unchanged storage after rejected conversions. Device
placement uses Atto's `gpu { ... }` execution surface with its own compiler
and device admission requirements.

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

## Host reference: shape v0 (not an Atto runtime implementation)

`reference/shape_rules.py` defines bounded shape validation, scalar/zero-axis element counts, row-major strides, trailing-axis broadcast compatibility and broadcast index mapping. It rejects booleans as dimensions, negative dimensions, rank over 64 and signed-64-bit element-count overflow. This establishes deterministic expected behavior for future Atto tensor backends; no CPU/GPU/SIMD Atto guest execution is claimed.

Run `python3 -m unittest discover -s tests -v` from the repository root to check the Python reference examples. Before porting, align error representation and resource bounds across compilers.
