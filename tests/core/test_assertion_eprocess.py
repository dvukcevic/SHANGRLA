import numpy as np
import pytest

from shangrla.core.Audit import Audit
from shangrla.core.NonnegMean import NonnegMean
from shangrla.core.assertion import Assertion, AssertionHandler, Assorter
from shangrla.core.cvr import CVR
from shangrla.core.eprocess import LeafNode
from shangrla.core.contest import Contest, Plurality


def make_handler():
    contest = Contest(
        id="mayor",
        risk_limit=0.05,
        cards=100,
        choice_function=Plurality(),
        candidates=["Alice", "Bob"],
        winner=["Alice"],
        audit_type=Audit.AUDIT_TYPE.POLLING,
        test=NonnegMean.alpha_mart,
        use_style=False,
    )
    return AssertionHandler(contest)


def make_sample(candidate, size):
    return [
        CVR(id=i, votes={"mayor": {candidate: True}})
        for i in range(size)
    ]


def test_empty_assertion_state_is_neutral_and_kwargs_are_not_shared():
    first = Assertion()
    second = Assertion()

    first.test_kwargs["option"] = True

    assert first.p_value() == 1
    assert first.min_p() == 1
    assert second.test_kwargs == {}


def test_min_p():
    node1 = LeafNode(id="first")
    node1.eprocess.append(np.log(1 / np.array([1, 0.5, 0.1, 0.01, 0.1])))
    node2 = LeafNode(id="second")
    node2.eprocess.append(np.log(1 / np.array([0.05])))
    asrtn1 = Assertion(node=node1)
    asrtn2 = Assertion(node=node2)

    assert asrtn1.min_p() == pytest.approx(0.01)
    assert asrtn2.min_p() == pytest.approx(0.05)


def test_compute_margin_does_not_shadow_numeric_margin():
    contest = Contest(
        id="mayor",
        choice_function=Plurality(),
        candidates=["Alice", "Bob"],
    )
    assertion = Assertion(
        contest=contest,
        assorter=Assorter(
            contest=contest,
            assort=lambda cvr: 1.0 if cvr.get_vote_for("mayor", "Alice") else 0.0,
        ),
        margin=0.25,
    )
    ballots = [
        CVR(id=1, votes={"mayor": {"Alice": True}}),
        CVR(id=2, votes={"mayor": {"Alice": True}}),
    ]

    assert assertion.compute_margin(ballots) == 1
    assert assertion.margin == 0.25


def test_set_p_values_updates_assertion_and_contest_summaries():
    handler = make_handler()
    sample = make_sample("Alice", 20)

    p_value = handler.set_p_values(sample)
    assertion_id, assertion = next(iter(handler.assertions.items()))

    assert np.isclose(p_value, assertion.p_value())
    assert handler.contest.p_values[assertion_id] == assertion.p_value()
    assert handler.contest.max_p == assertion.p_value()
    assert np.isclose(handler.registry.root.get_p_value(), assertion.p_value())
    assert assertion.proved
    assert handler.proved


def test_repeated_full_sample_update_recomputes_composition():
    handler = make_handler()
    sample = make_sample("Alice", 10)
    handler.set_p_values(sample)
    sample.extend(make_sample("Alice", 10))

    handler.set_p_values(sample)

    assertion = next(iter(handler.assertions.values()))
    assert len(assertion.node) == len(sample)
    assert len(handler.registry.root) == len(sample)
    assert np.isclose(
        handler.registry.root.get_p_value(),
        assertion.p_value(),
    )


def test_reset_clears_assertion_and_composed_evidence():
    handler = make_handler()
    handler.set_p_values(make_sample("Alice", 20))

    assert handler.reset_p_values()

    assertion_id, assertion = next(iter(handler.assertions.items()))
    assert len(assertion.node) == 0
    assert len(handler.registry.root) == 0
    assert assertion.p_value() == 1
    assert assertion.min_p() == 1
    assert not assertion.proved
    assert handler.contest.p_values == {assertion_id: 1.0}
    assert handler.contest.proved == {assertion_id: False}
    assert handler.contest.max_p == 1
    assert not handler.proved


def test_class_reset_resets_handlers_attached_to_contests():
    handler = make_handler()
    handler.set_p_values(make_sample("Alice", 20))
    handler.contest.assertions = handler

    assert Assertion.reset_p_values({"mayor": handler.contest})
    assert handler.contest.max_p == 1
