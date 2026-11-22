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

## IRV AWAIRE ballot-polling audit

The complete synthetic example is in
[`examples/workflow_irv_awaire_ballot_polling.py`](../examples/workflow_irv_awaire_ballot_polling.py).
An interactive starter is available at
[`examples/workflow_irv_awaire_ballot_polling_interactive.py`](../examples/workflow_irv_awaire_ballot_polling_interactive.py).
It configures the contest with `Contest.ASSERTION_STRATEGY.AWAIRE`, creates the
AWAIRE assertion tree, orders the sampled MVRs with
`CVR.prep_polling_sample`, and updates evidence with
`contest.assertions.set_p_values(mvr_sample)`. The sample demonstrates the
composed alternative-order process reaching certainty. The audit status is
reported with `audit.summarize_status(contests)`.

## IRV AWAIRE ballot-level comparison audit

The complete synthetic example is in
[`examples/workflow_irv_awaire_ballot_comparison.py`](../examples/workflow_irv_awaire_ballot_comparison.py).
An interactive starter is available at
[`examples/workflow_irv_awaire_ballot_comparison_interactive.py`](../examples/workflow_irv_awaire_ballot_comparison_interactive.py).
It configures the contest with `Contest.ASSERTION_STRATEGY.AWAIRE`, creates the
AWAIRE assertion tree, and calculates comparison margins from the reported
CVRs. The sample includes a reported-ranking error. The workflow prepares
paired MVR/CVR samples with `CVR.prep_comparison_sample`, then updates evidence
with `contest.assertions.set_p_values(mvr_sample, cvr_sample, use_all=True)`.
Some generated requirements may not be supported individually by the CVRs;
the audit result is determined by the composed alternative-order root process
reported by `audit.summarize_status(contests)`.
