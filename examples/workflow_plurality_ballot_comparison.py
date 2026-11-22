"""Workflow: end-to-end plurality ballot-level comparison audit."""

import numpy as np

from shangrla.core.Audit import Audit
from shangrla.core.NonnegMean import NonnegMean
from shangrla.core.assertion import make_all_assertions
from shangrla.core.contest import Contest, Plurality
from shangrla.core.cvr import CVR


def main() -> None:
    # Generate true ballots, then introduce one reported vote for the wrong candidate.
    rng = np.random.default_rng(2026)
    manual_votes = np.concatenate([
        np.full(900, "Alice", dtype="<U5"),
        np.full(700, "Bob", dtype="<U5"),
        np.full(400, "Candy", dtype="<U5"),
    ])
    rng.shuffle(manual_votes)
    reported_votes = manual_votes.copy()
    error_index = int(np.flatnonzero(manual_votes == "Bob")[0])
    reported_votes[error_index] = "Alice"

    # 1. Configure the audit and comparison contest.
    audit = Audit.from_dict({
        "seed": 93686630803205229070,
        "strata": {
            "single": {
                "max_cards": len(reported_votes),
                "use_style": False,
                "replacement": False,
                "audit_type": Audit.AUDIT_TYPE.CARD_COMPARISON,
            }
        },
    })
    contests = Contest.from_dict_of_dicts({
        "mayor": {
            "name": "Mayor",
            "risk_limit": 0.05,
            "cards": len(reported_votes),
            "choice_function": Plurality(),
            "n_winners": 1,
            "candidates": ["Alice", "Bob", "Candy"],
            "winner": ["Alice"],
            "audit_type": Audit.AUDIT_TYPE.CARD_COMPARISON,
            "test": NonnegMean.alpha_mart,
            "estim": NonnegMean.optimal_comparison,
            "use_style": False,
        }
    })
    audit.check_audit_parameters(contests)

    # 2. Create the winner-loser pair assertions to be tested.
    make_all_assertions(contests)

    # 3. Construct the reported CVRs and set comparison margins from their tally.
    cvrs = CVR.from_dict([
        {
            "id": str(index),
            "votes": {"mayor": {candidate: True}},
        }
        for index, candidate in enumerate(reported_votes)
    ])
    manual_cvrs = CVR.from_dict([
        {
            "id": str(index),
            "votes": {"mayor": {candidate: True}},
        }
        for index, candidate in enumerate(manual_votes)
    ])
    Contest.tally(contests, cvrs)
    contests["mayor"].assertions.set_all_margins_from_cvrs(audit, cvrs)

    # 4. Draw a sample that includes the known discrepancy and prepare matching MVRs.
    sample_indices = rng.choice(
        np.delete(np.arange(len(cvrs)), error_index), size=99, replace=False
    ).tolist()
    sample_indices.append(error_index)
    rng.shuffle(sample_indices)
    cvr_sample = [cvrs[index] for index in sample_indices]
    mvr_sample = [manual_cvrs[index] for index in sample_indices]
    sample_order = {
        str(index): {"selection_order": order}
        for order, index in enumerate(sample_indices)
    }
    CVR.prep_comparison_sample(mvr_sample, cvr_sample, sample_order)

    # 5. Update assertion evidence using each manual vote and its reported CVR.
    contests["mayor"].assertions.set_p_values(
        mvr_sample, cvr_sample, use_all=True
    )

    # 6. Report whether every assertion is below the contest risk limit.
    audit.summarize_status(contests)


if __name__ == "__main__":
    main()
