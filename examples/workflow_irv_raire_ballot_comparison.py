"""Workflow: end-to-end IRV RAIRE ballot-level comparison audit."""

import numpy as np

from shangrla.core.Audit import Audit
from shangrla.core.NonnegMean import NonnegMean
from shangrla.core.assertion import make_all_assertions
from shangrla.core.contest import Contest, InstantRunoff
from shangrla.core.cvr import CVR
from shangrla.raire.raire import compute_raire_assertions
from shangrla.raire.raire_utils import Contest as RaireContest
from shangrla.raire.sample_estimator import cp_estimate


def create_workflow_data() -> tuple[
    Audit, dict[str, Contest], list[CVR], list[CVR], list[CVR]
]:
    """Create a synthetic IRV contest, RAIRE assertions, and comparison sample."""
    rng = np.random.default_rng(2027)
    contest_id = "mayor"
    candidates = ["Alice", "Bob", "Candy", "Desert"]

    manual_rankings = (
        [("Alice", "Bob", "Candy")] * 20
        + [("Alice", "Bob", "Desert")] * 20
        + [("Bob",)] * 50
        + [("Candy", "Alice", "Bob")] * 12
        + [("Desert", "Alice", "Bob")] * 8
    )
    rng.shuffle(manual_rankings)

    reported_rankings = manual_rankings.copy()
    error_index = next(
        index
        for index, ranking in enumerate(manual_rankings)
        if ranking == ("Desert", "Alice", "Bob")
    )
    reported_rankings[error_index] = ("Desert", "Bob", "Alice")

    # RAIRE uses zero-based ranks; SHANGRLA CVRs use positive ranks.
    raire_cvrs = {
        str(index): {
            contest_id: {candidate: rank for rank, candidate in enumerate(ranking)}
        }
        for index, ranking in enumerate(reported_rankings)
    }
    raire_contest = RaireContest(
        contest_id, candidates, "Alice", len(reported_rankings)
    )
    raire_assertions = compute_raire_assertions(
        raire_contest, raire_cvrs, "Alice", cp_estimate, log=False
    )
    assertion_json = [assertion.to_json() for assertion in raire_assertions]

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
            "assertion_json": assertion_json,
            "audit_type": Audit.AUDIT_TYPE.CARD_COMPARISON,
            "test": NonnegMean.alpha_mart,
            "estim": NonnegMean.optimal_comparison,
            "use_style": False,
        }
    })
    audit.check_audit_parameters(contests)
    make_all_assertions(contests)

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

    sample_indices = rng.choice(
        np.delete(np.arange(len(cvrs)), error_index), size=24, replace=False
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
    """Calculate comparison evidence and report the audit status."""
    contests["mayor"].assertions.set_p_values(
        mvr_sample, cvr_sample, use_all=True
    )
    audit.summarize_status(contests)


def main() -> None:
    audit, contests, _, mvr_sample, cvr_sample = create_workflow_data()
    run_workflow(audit, contests, mvr_sample, cvr_sample)


if __name__ == "__main__":
    main()
