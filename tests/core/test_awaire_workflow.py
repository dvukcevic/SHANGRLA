import warnings

import numpy as np
import pytest

from shangrla.core.Audit import Audit
from shangrla.core.NonnegMean import NonnegMean
from shangrla.core.assertion import AssertionHandler, make_all_assertions
from shangrla.core.contest import Contest, InstantRunoff
from shangrla.core.cvr import CVR


def _contest_data(audit_type):
    rankings = (
        [("Alice", "Bob", "Candy")] * 4
        + [("Bob", "Alice", "Candy"), ("Candy", "Alice", "Bob")]
    )
    cvrs = CVR.from_dict([
        {
            "id": str(index),
            "votes": {
                "mayor": {
                    candidate: rank + 1
                    for rank, candidate in enumerate(ranking)
                }
            },
        }
        for index, ranking in enumerate(rankings)
    ])
    mvrs = CVR.from_dict([
        {
            "id": str(index),
            "votes": {
                "mayor": {
                    candidate: rank + 1
                    for rank, candidate in enumerate(ranking)
                }
            },
        }
        for index, ranking in enumerate(rankings)
    ])
    audit = Audit.from_dict({
        "strata": {
            "single": {
                "max_cards": len(cvrs),
                "use_style": False,
                "replacement": False,
                "audit_type": audit_type,
            }
        }
    })
    contests = Contest.from_dict_of_dicts({
        "mayor": {
            "name": "Mayor",
            "risk_limit": 0.05,
            "cards": len(cvrs),
            "choice_function": InstantRunoff(),
            "n_winners": 1,
            "candidates": ["Alice", "Bob", "Candy"],
            "winner": ["Alice"],
            "audit_type": audit_type,
            "assertion_strategy": Contest.ASSERTION_STRATEGY.AWAIRE,
            "test": NonnegMean.alpha_mart,
            "estim": (
                NonnegMean.optimal_comparison
                if audit_type == Audit.AUDIT_TYPE.CARD_COMPARISON
                else NonnegMean.shrink_trunc
            ),
            "use_style": False,
        }
    })
    return audit, contests, cvrs, mvrs


@pytest.mark.parametrize(
    "audit_type",
    [Audit.AUDIT_TYPE.POLLING, Audit.AUDIT_TYPE.CARD_COMPARISON],
)
def test_awaire_end_to_end_composes_infinite_evidence(audit_type):
    audit, contests, cvrs, mvrs = _contest_data(audit_type)
    audit.check_audit_parameters(contests)
    make_all_assertions(contests)
    handler = contests["mayor"].assertions

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        handler.set_all_margins_from_cvrs(audit, cvrs)

    sample_order = {
        cvr.id: {"selection_order": order}
        for order, cvr in enumerate(cvrs)
    }

    def update_evidence(sample_size):
        mvr_sample = mvrs[:sample_size]
        if audit_type == Audit.AUDIT_TYPE.POLLING:
            CVR.prep_polling_sample(mvr_sample, sample_order)
            handler.set_p_values(mvr_sample)
        else:
            cvr_sample = cvrs[:sample_size]
            CVR.prep_comparison_sample(mvr_sample, cvr_sample, sample_order)
            handler.set_p_values(mvr_sample, cvr_sample, use_all=True)

    update_evidence(3)
    update_evidence(len(mvrs))
    update_evidence(len(mvrs))

    assert any(
        np.isposinf(assertion.node.eprocess._values).any()
        for assertion in handler.assertions.values()
    )
    assert len(handler.registry.root) == len(mvrs)
    assert handler.registry.root.get_p_value() == 0


def test_awaire_strategy_is_explicit_even_if_raire_data_is_present():
    contest = Contest(
        id="mayor",
        choice_function=InstantRunoff(),
        candidates=["Alice", "Bob", "Candy"],
        winner=["Alice"],
        cards=100,
        test=NonnegMean.alpha_mart,
        assertion_strategy=Contest.ASSERTION_STRATEGY.AWAIRE,
        assertion_json=[{"not": "a RAIRE assertion"}],
    )

    handler = AssertionHandler(contest)

    assert any(" DND " in assertion_id for assertion_id in handler.assertions)
    assert any(" DB " in assertion_id for assertion_id in handler.assertions)


def test_unset_strategy_auto_selects_awaire_without_raire_payload():
    contest = Contest(
        id="mayor",
        choice_function=InstantRunoff(),
        candidates=["Alice", "Bob", "Candy"],
        winner=["Alice"],
        cards=100,
        test=NonnegMean.alpha_mart,
    )

    handler = AssertionHandler(contest)

    assert contest.assertion_strategy == Contest.ASSERTION_STRATEGY.AWAIRE
    assert any(" DND " in assertion_id for assertion_id in handler.assertions)


def test_unset_strategy_auto_selects_raire_when_payload_is_provided():
    contest = Contest(
        id="mayor",
        choice_function=InstantRunoff(),
        candidates=["Alice", "Bob", "Candy"],
        winner=["Alice"],
        cards=100,
        test=NonnegMean.alpha_mart,
        assertion_json=[{
            "winner": "Alice",
            "loser": "Bob",
            "assertion_type": "WINNER_ONLY",
        }],
    )

    handler = AssertionHandler(contest)

    assert contest.assertion_strategy == Contest.ASSERTION_STRATEGY.RAIRE
    assert set(handler.assertions) == {"Alice v Bob"}


def test_raire_strategy_requires_assertion_json():
    contest = Contest(
        id="mayor",
        choice_function=InstantRunoff(),
        candidates=["Alice", "Bob", "Candy"],
        winner=["Alice"],
        test=NonnegMean.alpha_mart,
        assertion_strategy=Contest.ASSERTION_STRATEGY.RAIRE,
    )

    with pytest.raises(ValueError, match="non-empty list"):
        AssertionHandler(contest)


@pytest.mark.parametrize(
    "assertion_json, message",
    [
        ([{"winner": "Alice", "loser": "Bob"}], "missing fields"),
        ([{
            "winner": "Alice",
            "loser": "Bob",
            "assertion_type": "UNKNOWN",
        }], "unsupported assertion_type"),
        ([{
            "winner": "Alice",
            "loser": "Bob",
            "assertion_type": "IRV_ELIMINATION",
            "already_eliminated": ["Alice"],
        }], "invalid already_eliminated"),
    ],
)
def test_auto_selected_raire_rejects_invalid_payload(assertion_json, message):
    contest = Contest(
        id="mayor",
        choice_function=InstantRunoff(),
        candidates=["Alice", "Bob", "Candy"],
        winner=["Alice"],
        test=NonnegMean.alpha_mart,
        assertion_json=assertion_json,
    )

    with pytest.raises(ValueError, match=message):
        AssertionHandler(contest)


def test_irv_builder_rejects_unknown_assertion_strategy():
    contest = Contest(
        id="mayor",
        choice_function=InstantRunoff(),
        candidates=["Alice", "Bob", "Candy"],
        winner=["Alice"],
        test=NonnegMean.alpha_mart,
        assertion_strategy="unknown",
    )

    with pytest.raises(ValueError, match="Unknown IRV assertion strategy"):
        AssertionHandler(contest)
