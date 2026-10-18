# P0 implementation / v1 contract reconciliation

Review branch is based on P0 implementation source commit `4218b73b9944634b459b6427ff3e397944f2aab2` (PR #1) and adds proposed v1 interface metadata from `7f081995072f76d5a435b0ee04763f05168b5fdc` (PR #3). All file transfers remain **inside the same Oreslang repository**.

**Actual implementation scope:** The P0 source has a List<float>-backed logical row-major tensor, not ownership-safe contiguous Array<T>. The v1 tensor CPU contract remains proposed, and cross-backend memory ownership/alias guarantees are not met.

The P0 implementation's `docs/CONTRACT.md` remains canonical for **what code currently does**; `docs/PROPOSED_INTEGRATION_V1.md` and `integration/contract.v1.json` state **future provider/consumer acceptance targets**. The contract `requires` set is an intended interface, **not** a validated import/link edge or a passed Oreslang compiler test.

To validate: run the Node contract test (metadata only) and then run `scripts/test.sh` with a pinned, executable, actual Oreslang compiler binary. Capture exact source and compiler Git revisions, fixture IDs, executed guest result, stdout, stderr, environment, and negative-case results. Never report a source-only or independent host oracle as native compiler conformance.

Private Git source transfer to an alternate GitHub organization remains blocked pending reviewed incident remediation at [conformance #22](https://github.com/ores-truffle-oreslang/oreslang-platform-conformance/issues/22) and protected source-org approval. No releases or merges are authorized by the metadata test alone.
