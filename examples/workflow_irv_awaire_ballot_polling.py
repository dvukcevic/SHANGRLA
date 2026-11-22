"""Workflow: end-to-end IRV AWAIRE ballot-polling audit."""

import numpy as np

from shangrla.core.Audit import Audit
from shangrla.core.NonnegMean import NonnegMean
from shangrla.core.assertion import make_all_assertions
from shangrla.core.contest import Contest, InstantRunoff
from shangrla.core.cvr import CVR


def create_workflow_data() -> tuple[
    Audit, dict[str, Contest], list[CVR], list[CVR]
]:
    """Create a synthetic IRV contest and ordered polling sample."""
    rng = np.random.default_rng(2027)
    contest_id = "mayor"
    candidates = ["Alice", "Bob", "Candy"]
    rankings = (
        [("Alice", "Bob", "Candy")] * 60
        + [("Bob", "Alice", "Candy")] * 25
        + [("Candy", "Alice", "Bob")] * 15
    )
    rng.shuffle(rankings)

    # 1. Configure the audit and the polling contest.
    audit = Audit.from_dict({
        "seed": 93686630803205229070,
        "strata": {
            "single": {
                "max_cards": len(rankings),
                "use_style": False,
                "replacement": False,
                "audit_type": Audit.AUDIT_TYPE.POLLING,
            }
        },
    })
    contests = Contest.from_dict_of_dicts({
        contest_id: {
            "name": "Mayor",
            "risk_limit": 0.05,
            "cards": len(rankings),
            "choice_function": InstantRunoff(),
            "n_winners": 1,
            "candidates": candidates,
            "winner": ["Alice"],
            "assertion_strategy": Contest.ASSERTION_STRATEGY.AWAIRE,
            "audit_type": Audit.AUDIT_TYPE.POLLING,
            "test": NonnegMean.alpha_mart,
            "estim": NonnegMean.shrink_trunc,
            "use_style": False,
        }
    })
    audit.check_audit_parameters(contests)

    # 2. Create the AWAIRE assertion tree to be tested.
    make_all_assertions(contests)

    # 3. Construct reported-ballot CVRs from the synthetic rankings.
    cvrs = CVR.from_dict([
        {
            "id": str(index),
            "votes": {
                contest_id: {
                    candidate: rank + 1
                    for rank, candidate in enumerate(ranking)
                }
            },
        }
        for index, ranking in enumerate(rankings)
    ])

    # 4. Draw a random sample and put it in polling-selection order.
    # This decisive synthetic sample illustrates certainty in the composed
    # AWAIRE alternative-order process.
    sample_indices = rng.choice(len(cvrs), size=90, replace=False).tolist()
    mvr_sample = [cvrs[index] for index in sample_indices]
    CVR.prep_polling_sample(mvr_sample, {
        str(index): {"selection_order": order}
        for order, index in enumerate(sample_indices)
    })
    return audit, contests, cvrs, mvr_sample


def run_workflow(
    audit: Audit,
    contests: dict[str, Contest],
    mvr_sample: list[CVR],
) -> None:
    """Calculate polling evidence and report composed audit status."""

    # 5. Update the assertion evidence with the manual vote records.
    contests["mayor"].assertions.set_p_values(mvr_sample)

    # 6. Report whether the composed alternative-order process is conclusive.
    audit.summarize_status(contests)


def main() -> None:
    audit, contests, _, mvr_sample = create_workflow_data()
    run_workflow(audit, contests, mvr_sample)


if __name__ == "__main__":
    main()
