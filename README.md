> **AI agent instructions (organization-wide):** Before working on code, issues, or pull requests, read and follow the [organization `AGENTS.md`](https://github.com/ores-truffle-oreslang/.github/blob/main/AGENTS.md), including its **merge-first stealth-development policy**. Honor any repository-local `AGENTS.md`, explicit holds, and GitHub-enforced rules.

# Oreslang Tensor

## AttoLang migration and authoritative specification (2026-11-18)

**AttoLang** is the current public name of Oreslang. This `ores-truffle-oreslang/oreslang-tensor` repository retains its historical context. **Current AttoLang destination:** [atto-lang/atto-tensor](https://github.com/atto-lang/atto-tensor). Legacy source and commands below reflect this checkout; do not automatically substitute new CLI names or claim feature parity. These repositories are not assumed to have identical Git histories, issues, pull requests, or implementation results.

**The canonical language specification is [atto-lang/atto-spec](https://github.com/atto-lang/atto-spec)**. Portable `.atto` modules belong in [atto-common-source](https://github.com/atto-lang/atto-common-source) and executed cross-backend positive/negative fixtures in [atto-platform-conformance](https://github.com/atto-lang/atto-platform-conformance). The proposed modern rules include `snake_case` methods, ownership `borrow!`/`take!`/`share!`/`copy!`/`copy!!`, [implicit borrowing](https://github.com/atto-lang/atto-spec/blob/main/language/implicit-argument-borrow-v1.md), [file/module constants including nonthrowing launch-time environment reads](https://github.com/atto-lang/atto-spec/blob/main/language/file-module-const-v1.md), and [separate bitwise-shift and directional-stream operators](https://github.com/atto-lang/atto-spec/blob/main/language/bitwise-and-stream-operators-v1.md). These are contract/design directions, not a claim this legacy implementation supports all syntax.

Preserve historic `.ores` paths, command names, ABI/wire names and pinned fixtures until verified replacements exist. Avoid Python support tooling/tests; prefer Rust. Do not infer successful compilation, actor isolation or cross-backend parity from README examples, host-only tests, or zero-step GitHub Actions. Document exact commit/tree, target, toolchain, executed guest tests and explicit failures/unsupported status.

Native Oreslang tensor semantics and CPU reference kernels. The first tranche uses plain `.ores` source and `List<T>` backing storage; this is not yet an ownership-safe contiguous `Array<T>` implementation or an accelerated backend.

## Scope (P0)
- Checked shape/product and contiguous tensor descriptors.
- Scalar reference kernels with caller-supplied outputs.
- No Python/Java numerical implementation in public tensor algorithms.
- Unsafe aliases, unsupported devices, overflow and shape mismatches must fail closed.

See `docs/CONTRACT.md` and `src/tensor.ores` after bootstrap PR.

## No-Python policy

Python source, scripts, tests, notebooks, package manifests, runtime dependencies, and CI tooling are not permitted in this repository. Host-side tooling and reference tests must be implemented in Rust with equivalent behavior and coverage. Documentation may mention Python only to describe this policy or historical migration work.
