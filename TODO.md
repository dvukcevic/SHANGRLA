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
- [ ] Verify that AWAIRE-style requirements are implemented correctly using
      assertions and combiners, then delete `requirement.py`.
- [ ] Document the desired end-to-end workflow for a single contest and
      implement any missing functionality needed to support it.
- [ ] Review `shangrla/core/__init__.py` and improve its package exports and
      import organization.
