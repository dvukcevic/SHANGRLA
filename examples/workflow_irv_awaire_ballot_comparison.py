"""Workflow: end-to-end IRV AWAIRE ballot-level comparison audit."""

import numpy as np

from shangrla.core.Audit import Audit
from shangrla.core.NonnegMean import NonnegMean
from shangrla.core.assertion import make_all_assertions
from shangrla.core.contest import Contest, InstantRunoff
from shangrla.core.cvr import CVR


def create_workflow_data() -> tuple[
    Audit, dict[str, Contest], list[CVR], list[CVR], list[CVR]
]:
    """Create an IRV contest, a reported CVR error, and paired comparison samples."""
    rng = np.random.default_rng(2027)
    contest_id = "mayor"
    candidates = ["Alice", "Bob", "Candy"]
    manual_rankings = (
        [("Alice", "Bob", "Candy")] * 60
        + [("Bob", "Alice", "Candy")] * 25
        + [("Candy", "Alice", "Bob")] * 15
    )
    rng.shuffle(manual_rankings)
    reported_rankings = manual_rankings.copy()
    error_index = next(
        index
        for index, ranking in enumerate(manual_rankings)
        if ranking[0] == "Alice"
    )
    reported_rankings[error_index] = ("Bob", "Alice", "Candy")

    # 1. Configure the audit and the comparison contest.
    audit = Audit.from_dict({
        "seed": 93686630803205229070,
        "strata": {
            "single": {
                "max_cards": len(reported_rankings),
                "use_style": False,
                "replacement": False,
                "audit_type": Audit.AUDIT_TYPE.CARD_COMPARISON,
            }
        },
    })
    contests = Contest.from_dict_of_dicts({
        contest_id: {
            "name": "Mayor",
            "risk_limit": 0.05,
            "cards": len(reported_rankings),
            "choice_function": InstantRunoff(),
            "n_winners": 1,
            "candidates": candidates,
            "winner": ["Alice"],
            "assertion_strategy": Contest.ASSERTION_STRATEGY.AWAIRE,
            "audit_type": Audit.AUDIT_TYPE.CARD_COMPARISON,
            "test": NonnegMean.alpha_mart,
            "estim": NonnegMean.optimal_comparison,
            "use_style": False,
        }
    })
    audit.check_audit_parameters(contests)

    # 2. Create the AWAIRE assertion tree to be tested.
    make_all_assertions(contests)

    # 3. Construct reported CVRs, matching manual CVRs, and comparison margins.
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
        for index, ranking in enumerate(reported_rankings)
    ])
    manual_cvrs = CVR.from_dict([
        {
            "id": str(index),
            "votes": {
                contest_id: {
                    candidate: rank + 1
                    for rank, candidate in enumerate(ranking)
                }
            },
        }
        for index, ranking in enumerate(manual_rankings)
    ])
    contests[contest_id].assertions.set_all_margins_from_cvrs(audit, cvrs)

    # 4. Draw a paired sample and put both lists in comparison-selection order.
    # AWAIRE leaf assertions are components of alternative-order branches;
    # status is determined by the composed root, not each leaf in isolation.
    sample_indices = rng.choice(
        [index for index in range(len(cvrs)) if index != error_index],
        size=39,
        replace=False,
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
    return audit, contests, cvrs, mvr_sample, cvr_sample


def run_workflow(
    audit: Audit,
    contests: dict[str, Contest],
    mvr_sample: list[CVR],
    cvr_sample: list[CVR],
) -> None:
    """Calculate comparison evidence and report composed audit status."""

    # 5. Update all AWAIRE leaf evidence with the paired manual/reported votes.
    contests["mayor"].assertions.set_p_values(
        mvr_sample, cvr_sample, use_all=True
    )

    # 6. Report whether the composed alternative-order root is conclusive.
    audit.summarize_status(contests)


def main() -> None:
    audit, contests, _, mvr_sample, cvr_sample = create_workflow_data()
    run_workflow(audit, contests, mvr_sample, cvr_sample)


if __name__ == "__main__":
    main()
