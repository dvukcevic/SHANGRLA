"""Workflow: end-to-end plurality ballot-polling audit."""

import numpy as np

from shangrla.core.Audit import Audit
from shangrla.core.NonnegMean import NonnegMean
from shangrla.core.assertion import make_all_assertions
from shangrla.core.contest import Contest, Plurality
from shangrla.core.cvr import CVR


def create_workflow_data() -> tuple[Audit, dict[str, Contest], list[CVR], list[CVR]]:
    """Create the audit setup and polling sample for interactive inspection."""
    # Generate a synthetic reported tally for a single plurality contest.
    rng = np.random.default_rng(2025)
    ballots = np.concatenate([
        np.full(900, "Alice", dtype="<U5"),
        np.full(700, "Bob", dtype="<U5"),
        np.full(400, "Candy", dtype="<U5"),
    ])
    rng.shuffle(ballots)

    # 1. Configure the audit and the polling contest.
    audit = Audit.from_dict({
        "seed": 93686630803205229070,
        "strata": {
            "single": {
                "max_cards": len(ballots),
                "use_style": False,
                "replacement": False,
                "audit_type": Audit.AUDIT_TYPE.POLLING,
            }
        },
    })
    contests = Contest.from_dict_of_dicts({
        "mayor": {
            "name": "Mayor",
            "risk_limit": 0.05,
            "cards": len(ballots),
            "choice_function": Plurality(),
            "n_winners": 1,
            "candidates": ["Alice", "Bob", "Candy"],
            "winner": ["Alice"],
            "audit_type": Audit.AUDIT_TYPE.POLLING,
            "test": NonnegMean.alpha_mart,
            "estim": NonnegMean.shrink_trunc,
            "use_style": False,
            # Explicitly set shrink_trunc parameters:
            "eta": 0.55,  # default: u*(1-epsilon) = 0.999
            "d": 20,  # default: 100
        }
    })
    audit.check_audit_parameters(contests)

    # 2. Create the winner-loser pair assertions to be tested.
    make_all_assertions(contests)

    # 3. Construct the reported-ballot CVRs, tally them, and set assorter margins.
    cvrs = CVR.from_dict([
        {
            "id": str(index),
            "votes": {"mayor": {candidate: True}},
        }
        for index, candidate in enumerate(ballots)
    ])
    Contest.tally(contests, cvrs)
    contests["mayor"].find_margins_from_tally()

    # 4. Draw a random sample and record the ballots as manual vote records.
    sample = rng.choice(cvrs, size=100, replace=False).tolist()
    CVR.prep_polling_sample(sample, {
        cvr.id: {"selection_order": index}
        for index, cvr in enumerate(sample)
    })

    return audit, contests, cvrs, sample


def run_workflow(
    audit: Audit,
    contests: dict[str, Contest],
    sample: list[CVR],
) -> None:
    """Calculate polling evidence and report the audit status."""
    # 5. Update the assertion evidence with the manual vote records.
    contests["mayor"].assertions.set_p_values(sample)

    # 6. Report whether every assertion is below the contest risk limit.
    audit.summarize_status(contests)


def main() -> None:
    audit, contests, _, sample = create_workflow_data()
    run_workflow(audit, contests, sample)


if __name__ == "__main__":
    main()
