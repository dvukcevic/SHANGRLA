# Workflows

This document collects end-to-end workflows for conducting some example
single-contest audits using SHANGRLA.

IRV contests can select assertion construction with
`Contest.assertion_strategy`. If it is unset, SHANGRLA uses RAIRE when a
non-`None` `assertion_json` payload is present, and AWAIRE otherwise. Set
`Contest.ASSERTION_STRATEGY.AWAIRE` or `Contest.ASSERTION_STRATEGY.RAIRE`
explicitly to override auto-selection. A selected RAIRE strategy requires a
non-empty, valid `assertion_json` payload.

## Plurality ballot-polling audit

The complete synthetic example is in
[`examples/workflow_plurality_ballot_polling.py`](../examples/workflow_plurality_ballot_polling.py).

The workflow is:

1. Create one `Audit` and one `Contest`, with `Audit.AUDIT_TYPE.POLLING`.
2. Create the contest assertions with `make_all_assertions(contests)`.
3. Construct CVRs for the reported ballots, tally them, and set assertion margins
   with `Contest.tally(contests, cvrs)` followed by
   `contests["mayor"].find_margins_from_tally()`.
4. Draw a random sample of physical ballots, record each result as a `CVR`, and
   put the MVR list in sample-selection order.
5. Call `contest.assertions.set_p_values(mvr_sample)` after each set of
   manual results.
6. Call `audit.summarize_status(contests)` to determine whether all assertions
   have crossed the risk limit or whether the audit needs more ballots.

## Plurality ballot-level comparison audit

The complete synthetic example is in
[`examples/workflow_plurality_ballot_comparison.py`](../examples/workflow_plurality_ballot_comparison.py).
It includes one reported vote for Alice that the manual examination corrects to
a vote for Bob.

The workflow is:

1. Create one `Audit` and one `Contest`, with
   `Audit.AUDIT_TYPE.CARD_COMPARISON`.
2. Create the contest assertions with `make_all_assertions(contests)`.
3. Construct reported CVRs, tally them, and set each assertion's comparison
   margin with `set_all_margins_from_cvrs(audit, cvrs)`.
4. Draw a random sample of CVRs and create matching MVRs, including each
   ballot's manually interpreted vote.
5. Call `CVR.prep_comparison_sample(mvr_sample, cvr_sample, sample_order)` to
   put both lists in the same selection order, then update assertion evidence
   with `contest.assertions.set_p_values(mvr_sample, cvr_sample, use_all=True)`.
6. Call `audit.summarize_status(contests)` to determine whether all assertions
   have crossed the risk limit or whether the audit needs more ballots.

## IRV RAIRE ballot-level comparison audit

The complete synthetic example is in
[`examples/workflow_irv_raire_ballot_comparison.py`](../examples/workflow_irv_raire_ballot_comparison.py).

The workflow is:

1. Construct a RAIRE contest and use `compute_raire_assertions` to generate
   the assertion JSON for the reported winner.
2. Create one `Audit` and one `Contest`, with
   `Audit.AUDIT_TYPE.CARD_COMPARISON` and
   `Contest.ASSERTION_STRATEGY.RAIRE`. Pass the generated JSON as
   `assertion_json`; an explicit RAIRE strategy requires non-empty, valid
   assertion JSON.
3. Create the SHANGRLA assertions with `make_all_assertions(contests)`.
4. Construct the reported CVRs and matching manually interpreted MVRs. RAIRE
   rankings use zero-based ranks, while SHANGRLA CVRs use positive ranks.
5. Set comparison margins from the reported CVRs with
   `contest.assertions.set_all_margins_from_cvrs(audit, cvrs)`.
6. Draw a random paired sample and call
   `CVR.prep_comparison_sample(mvr_sample, cvr_sample, sample_order)` to put
   both lists in selection order.
7. Call `contest.assertions.set_p_values(mvr_sample, cvr_sample, use_all=True)`
   to update the evidence for every RAIRE assertion.
8. Call `audit.summarize_status(contests)` to determine whether the RAIRE
   assertion set has reached the risk limit or needs more ballots.

## IRV AWAIRE ballot-polling audit

The complete synthetic example is in
[`examples/workflow_irv_awaire_ballot_polling.py`](../examples/workflow_irv_awaire_ballot_polling.py).

The workflow is:

1. Create one `Audit` and one `Contest`, with `Audit.AUDIT_TYPE.POLLING` and
   `Contest.ASSERTION_STRATEGY.AWAIRE`.
2. Create the AWAIRE assertion tree with `make_all_assertions(contests)`.
3. Construct the reported-ballot CVRs from the synthetic rankings.
4. Draw a random sample of CVRs as MVRs and put the polling sample in
   selection order with `CVR.prep_polling_sample`.
5. Call `contest.assertions.set_p_values(mvr_sample)` to update the composed
   alternative-order evidence.
6. Call `audit.summarize_status(contests)` to report whether the composed
   process has reached certainty or requires more ballots.

## IRV AWAIRE ballot-level comparison audit

The complete synthetic example is in
[`examples/workflow_irv_awaire_ballot_comparison.py`](../examples/workflow_irv_awaire_ballot_comparison.py).

The workflow is:

1. Create one `Audit` and one `Contest`, with
   `Audit.AUDIT_TYPE.CARD_COMPARISON` and
   `Contest.ASSERTION_STRATEGY.AWAIRE`.
2. Create the AWAIRE assertion tree with `make_all_assertions(contests)`.
3. Construct reported CVRs and matching manually interpreted MVRs, including
   the reported-ranking error, and set comparison margins with
   `set_all_margins_from_cvrs(audit, cvrs)`.
4. Draw a random paired sample and call
   `CVR.prep_comparison_sample(mvr_sample, cvr_sample, sample_order)` to put
   both lists in selection order.
5. Call `contest.assertions.set_p_values(mvr_sample, cvr_sample, use_all=True)`
   to update all AWAIRE leaf evidence.
6. Call `audit.summarize_status(contests)` to determine the audit result from
   the composed alternative-order root process.
