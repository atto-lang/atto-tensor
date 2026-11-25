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

## Organization boundaries and legacy compatibility

Tensor owns checked layouts and kernels, not production model serving or inference billing. Preserve the existing host-reference checks while migrating tooling to Rust; do not introduce a host-language numerical implementation as the public Atto tensor API. New portable language libraries, compilers, tooling and conformance belong in AttoLang; managed cloud hosting, deployment, tenant identity and billing belong in separately governed services. Preserve supported legacy .ores source and historical package/CLI aliases until compiler-backed .atto replacements and consumers are validated. Similar repository names do not establish an identity-preserving transfer or source parity.

Any future repository transfer requires an approved owner and rollback plan, independently authorized source and destination, recorded numeric repository IDs and commit/tree, verified visibility, rulesets, access, releases, packages, consumers, webhooks, Apps, secrets and runner continuity. Never transfer private source without separate explicit authorization. Verify identity, history and controls after a transfer and rerun exact-head CI with executed steps; zero-step and host-only tests do not certify native guest execution.
