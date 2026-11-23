# TODO

- [x] Make legacy test processes special cases of `alpha_mart`.
- [x] Handle certainty and infinite evidence, including verifying that evidence is
      returned as `inf` only when the result is certain.
- [x] Cap converted p-values at 1.
- [x] Thoroughly test `MinOfRunningMax` and verify that retaining only partial
      history still produces correct calculations.
- [x] Move CVR functionality into a separate module, apart from contest and
      social-choice-function handling.
- [x] Check for direct access to class properties where a method should be used.
- [x] Review the CVR implementation for gaps and add the missing tests.
- [x] Set the minimum supported Python version to 3.12 and update references
      that still specify an older version.
- [x] Verify that AWAIRE-style requirements are implemented correctly using
      assertions and combiners, then delete `requirement.py`.
- [x] Fix issues to allow AWAIRE to work:
  - [x] Preserve positive infinite log evidence when recomputing composed node
        processes, while rejecting NaN and negative-infinite inputs.
  - [x] Add end-to-end tests for AWAIRE ballot-polling and ballot-level
        comparison, including evidence updates and node-tree recomputation.
  - [x] Replace the heuristic that selects RAIRE versus AWAIRE based on whether
        `assertion_json` exists with an explicit assertion-strategy indicator.
- [x] Document the desired end-to-end workflow for a variety of single contests
      and implement any missing functionality needed to support it:
  - [x] Plurality ballot-polling
  - [x] Plurality ballot-level comparison
  - [x] IRV RAIRE ballot-level comparison
  - [x] IRV AWAIRE ballot-polling
  - [x] IRV AWAIRE ballot-level comparison
- [ ] Change the implementation so that e-values are calculated and stored
      directly, rather than going through a conversion into p-values.  The only
      such conversion should happen 'at the end' when a value needs to be
      reported or to determine if a stopping condition has been met.
- [ ] Review `shangrla/core/__init__.py` and improve its package exports and
      import organization.
- [ ] Review the use of IRV 'assertion strategies' and try to unify them, so
      that we can use both RAIRE and AWAIRE together.
- [ ] Implement AWAIRE v2 ('incremental AWAIRE'), to avoid exhaustive
      enumeration.
