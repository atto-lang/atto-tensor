# Oreslang Tensor

Native Oreslang tensor semantics and CPU reference kernels. The first tranche uses plain `.ores` source and `List<T>` backing storage; this is not yet an ownership-safe contiguous `Array<T>` implementation or an accelerated backend.

## Scope (P0)
- Checked shape/product and contiguous tensor descriptors.
- Scalar reference kernels with caller-supplied outputs.
- No Python/Java numerical implementation in public tensor algorithms.
- Unsafe aliases, unsupported devices, overflow and shape mismatches must fail closed.

See `docs/CONTRACT.md` and `src/tensor.ores` after bootstrap PR.

## No-Python policy

Python source, scripts, tests, notebooks, package manifests, runtime dependencies, and CI tooling are not permitted in this repository. Host-side tooling and reference tests must be implemented in Rust with equivalent behavior and coverage. Documentation may mention Python only to describe this policy or historical migration work.

## Repository ownership and migration (2026-11-08)

**AttoLang** is the public language, formerly Oreslang. The repository remains `ores-truffle-oreslang/oreslang-tensor` in the legacy org until approved cutover. Preserve supported `.ores` sources, old import paths and package/CLI names; don't infer backend completeness from a README.

- **Belongs here:** AttoLang tensor structures, layout algorithms and deterministic numerical tests.
- **Excluded:** production model serving and commercial inference billing.
- **Organization split:** [atto-lang](https://github.com/atto-lang) owns portable core/std, compilers, language libraries, test harnesses and demos; [atto-co](https://github.com/atto-co) independently owns cloud hosting, deployment, identity, billing and managed services. Existing `atto-*` repos are not automatically migrated copies or equivalent codebases.
- **Proposed first transfer (not executed):** retain original name, repo ID and Git history in `atto-lang/oreslang-tensor` via approved GitHub ownership transfer. No direct source-org → `atto-co` moves are authorized currently; future product carve-outs need separate signoffs.

**Maintainer safety procedure:** coordinate exact source head, owners and rollback in [issue #26](https://github.com/ores-truffle-oreslang/.github/issues/26), then follow private [runbook draft #25](https://github.com/ores-truffle-oreslang/.github/pull/25). Obtain source/target admin and independent security approval; inventory exact SHA/tree, numeric ID, target branch protections/rulesets, access, open PRs, releases/tags, packages, consumers, Actions/runners, secrets, GitHub Apps, webhooks and Pages. Don't clone/mirror private code as transfer workaround. After transfer verify ID/tree/history/controls, update links and rerun exact-head CI; **`steps=[]`/`null` is `NOT_EXECUTED`**, and host-only/source-free checks are not native guest/compiler evidence.
