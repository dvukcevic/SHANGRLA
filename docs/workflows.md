# Workflows

This document collects end-to-end workflows for conducting and extending
SHANGRLA audits.

## Single-contest plurality ballot-polling audit

The complete synthetic example is
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
