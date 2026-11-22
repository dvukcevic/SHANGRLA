from __future__ import annotations

from dataclasses import dataclass
import math
from abc import ABC, abstractmethod

import numpy as np
import json
import csv
import types
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Collection, Iterable
from typing import Tuple, Callable
from cryptorandom.cryptorandom import int_from_hash
from cryptorandom.sample import random_permutation
from cryptorandom.sample import sample_by_index

from .NonnegMean import NonnegMean
from .Audit import Audit
from .cvr import CVR


@dataclass
class SocialChoiceFunction(ABC):
    n_winners: int = 1
    winners: set[str] = None

    # @abstractmethod
    # def tabulate(self, votes: Collection[CVR], candidates: list[str]) -> Tabulation:
    #     """
    #     Tabulate the votes according to the social choice function.
    #     """
    #     pass

    def assort_upper_bound(self):
        return 1

    @abstractmethod
    def assort(self,
               c: CVR,
               contest_id,
               winner: str,
               loser: str,
               winner_cands: list[str],
               loser_cands: list[str] = None
               ) -> float:
        pass

    # @abstractmethod
    # def get_winner_loser_pairs(self,
    #                            winners: list[str],
    #                            losers: list[str]
    #                            ) -> Iterable[tuple[str, str, str, list[str], list[str]]]:
    #     """
    #     Get the winner-loser pairs for the social choice function.
    #
    #     Parameters
    #     ----------
    #     winners: list[str]
    #         List of winners.
    #     losers: list[str]
    #         List of losers.
    #
    #     Returns
    #     -------
    #     Iterable[tuple[str, str, str, list[str], list[str]]]:
    #         Iterable of tuples containing the name, winner, loser, winner_candidates, loser_candidates.
    #         winner_/loser_candidates mean the set of candidates under consideration when evaluating the winner/loser
    #     """
    #     pass


class Plurality(SocialChoiceFunction):
    def assort(self,
               c: CVR,
               contest_id,
               winner: str,
               loser: str,
               winner_cands: list[str],
               loser_cands: list[str] = None
               ) -> float:
        if loser_cands is None:
            loser_cands = winner_cands
        winr = CVR.as_vote(c.get_vote_for(contest_id, winner)) if winner in winner_cands else 0
        losr = CVR.as_vote(c.get_vote_for(contest_id, loser)) if loser in loser_cands else 0
        return (winr - losr + 1) / 2

    def get_winner_loser_pairs(self,
                              winners: list[str],
                              losers: list[str]
                              ) -> Iterable[tuple[str, str, list[str], list[str]]]:
        cands = winners + losers
        for winner in winners:
            for loser in losers:
                yield f"{winner} v {loser}", winner, loser, cands, cands


@dataclass
class Threshold(SocialChoiceFunction):
    threshold: float = 2/3

    def assort_upper_bound(self):
        return 1 / (2 * self.threshold)

    def assort(self, c: CVR, contest_id, winner: str, loser: str,
               winner_cands: list[str], loser_cands: list[str] = None) -> float:
        winr = CVR.as_vote(c.get_vote_for(contest_id, winner))
        return winr / (2 * self.threshold) if c.has_one_vote(contest_id, winner_cands) else 1 / 2

    def get_winner_loser_pairs(self,
                              winners: list[str],
                              losers: list[str]
                              ) -> Iterable[tuple[str, str, str, list[str], list[str]]]:
        cands = winners + losers
        for winner in winners:
            for loser in losers:
                yield winner, loser, cands, cands


Supermajority = Threshold


class Approval(SocialChoiceFunction):
    pass


class DHont(SocialChoiceFunction):
    pass


@dataclass
class InstantRunoff(SocialChoiceFunction):
    max_vote_length: int = np.inf  # TODO: use this to truncate/eliminate CVRs that exceeds max length?

    def assort(self, c: CVR, contest_id, winner: str, loser: str,
               winner_cands: list[str], loser_cands: list[str] = None) -> float:
        if loser_cands is None:
            loser_cands = winner_cands

        # Ordinary IRV assertions compare two candidates across equal
        # remaining-candidate sets.  These include WINNER_ONLY assertions
        # and IRV_ELIMINATION assertions with an explicit set of candidates
        # eliminated before the comparison.  The code at the bottom of this
        # function handles those cases.

        # This section of code handles AWAIRE DND assertions (requirements),
        # which compare two candidates across unequal remaining-candidate
        # sets.
        #
        # An AWAIRE DND requirement is asymmetric: the first candidate must
        # never be eliminated while testing the second candidate.
        if (
            loser_cands is not None
            and len(loser_cands) == 2
            and set(loser_cands) == {winner, loser}
            and set(winner_cands) != set(loser_cands)
        ):
            rank_winner = c.get_vote_for(contest_id, winner)
            rank_loser = c.get_vote_for(contest_id, loser)
            if rank_winner and (
                not rank_loser or rank_winner < rank_loser
            ):
                for candidate in winner_cands:
                    if candidate in (winner, loser):
                        continue
                    rank_candidate = c.get_vote_for(contest_id, candidate)
                    if rank_candidate and rank_candidate < rank_winner:
                        if rank_loser:
                            return 0
                        return 1 / 2
                return 1
            if rank_loser:
                return 0
            return 1 / 2

        # The following code handles ordinary IRV assertions, which
        # compare two candidates across equal remaining-candidate sets.
        # See earlier comments, above.
        winr = c.rcv_votefor_cand(contest_id, winner, winner_cands)
        losr = c.rcv_votefor_cand(contest_id, loser, loser_cands)
        return (winr - losr + 1) / 2


class Contest:
    """
    Objects and methods for contests.
    """

    class CANDIDATES:
        """
        constants for referring to candidates and candidate groups.

        For example, in a supermajority contest where no candidate is reported to have won,
        the winner is Contest.CANDIDATES.NO_CANDIDATE, and in a supermajority contest in which one
        candidate is reported to have won, the loser (for the assorter) is Contest.CANDIDATES.ALL_OTHERS
        """

        CANDIDATES = (
            ALL := "ALL",
            ALL_OTHERS := "ALL_OTHERS",
            WRITE_IN := "WRITE_IN",
            NO_CANDIDATE := "NO_CANDIDATE",
        )

    ATTRIBUTES = (
        "id",
        "name",
        "risk_limit",
        "cards",
        "choice_function",
        "n_winners",
        "share_to_win",
        "candidates",
        "winner",
        "assertion_file",
        "audit_type",
        "test",
        "test_kwargs",
        "g",
        "use_style",
        "assertions",
        "tally",
        "sample_size",
        "sample_threshold",
    )

    def __init__(
            self,
            id: object = None,
            name: str = None,
            risk_limit: float = 0.05,
            cards: int = 0,
            choice_function: SocialChoiceFunction = Plurality,
            n_winners: int = 1,
            share_to_win: float = None,
            candidates: Collection = None,
            winner: Collection = None,
            assertion_file: str = None,
            audit_type: str = Audit.AUDIT_TYPE.CARD_COMPARISON,
            test: callable = None,
            test_kwargs: dict = {},
            g: float = 0.1,
            estim: callable = None,
            bet: callable = None,
            use_style: bool = True,
            assertions: dict = None,
            tally: dict = None,
            sample_size: int = None,
            sample_threshold: float = None,
            **kwargs,
    ):
        self.id = id
        self.name = name
        self.risk_limit = risk_limit
        self.cards = cards
        self.choice_function = choice_function
        self.n_winners = n_winners
        self.share_to_win = share_to_win
        self.candidates = candidates
        self.winner = winner
        self.assertion_file = assertion_file
        self.audit_type = audit_type
        self.test = test
        self.test_kwargs = test_kwargs
        self.g = g
        self.estim = estim
        self.bet = bet
        self.use_style = use_style
        self.assertions = assertions
        self.tally = tally
        self.sample_size = sample_size
        self.sample_threshold = sample_threshold
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)

    def find_sample_size(
            self,
            audit: object = None,
            mvr_sample: list = None,
            cvr_sample: Collection = None,
            **kwargs,
    ) -> int:
        """
        Estimate the sample size required to confirm the contest at its risk limit.

        This function can be used with or without data, for Audit.AUDIT_TYPE.POLLING,
        Audit.AUDIT_TYPE.CARD_COMPARISON, and Audit.AUDIT_TYPE.ONEAUDIT audits.

        The simulations in this implementation are inefficient because the randomization happens separately
        for every assorter, rather than in parallel.

        Parameters
        ----------
        cvrs: list of CVRs
            data (or simulated data) to base the sample size estimates on
        mvrs: list of MVRs (CVR objects)
            manually read votes to base the sample size estimates on, if data are available.

        Returns
        -------
        estimated sample size

        Side effects
        ------------
        sets self.sample_size to the estimated sample size

        """
        self.sample_size = 0
        for a in self.assertions.values():
            data = None
            if mvr_sample is not None:  # process the MVRs/CVRs to get data appropriate to each assertion
                data, u = a.mvrs_to_data(mvr_sample, cvr_sample)
            elif self.audit_type == Audit.AUDIT_TYPE.ONEAUDIT:
                data, u = a.mvrs_to_data(cvr_sample, cvr_sample)
            self.sample_size = max(
                self.sample_size,
                a.find_sample_size(
                    data=data,
                    rate_1=audit.error_rate_1,
                    rate_2=audit.error_rate_2,
                    reps=audit.reps,
                    quantile=audit.quantile,
                    seed=audit.sim_seed,
                ),
            )
        return self.sample_size

    def find_margins_from_tally(self):
        """
        Use the `Contest.tally` attribute to set the margins of the contest's assorters.

        Appropriate only for the social choice functions
                Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                Contest.SOCIAL_CHOICE_FUNCTION.APPROVAL


        Parameters
        ----------
        None

        Returns
        -------
        None

        Side effects
        ------------
        sets Assertion.margin for all Assertions in the Contest
        """
        for a, assn in self.assertions.items():
            assn.find_margin_from_tally()

    def make_assort_function(self, winner: str, loser: str | None,
                             winner_cands: list[str] = None, loser_cands: list[str] = None) -> Callable[[CVR], float]:
        if winner_cands is None:
            winner_cands = list(self.candidates)
        return lambda v: self.choice_function.assort(v, contest_id=self.id, winner=winner, loser=loser,
                                                       winner_cands=winner_cands, loser_cands=loser_cands)

    @classmethod
    def check_cards(cls, contests: Collection['Contest'], cvrs: Collection[CVR], force: bool = False):
        '''
        Check whether the number of CVRs that contain each contest is not greater than the upper bound
        on the number of cards that contain the contest; optionally, increase the upper bounds to make that so.

        Parameters
        ----------
        contests: collection of Contests

        cvrs: collection of CVRs

        force: bool
            Increase the upper bounds to include all the CVRs.
            This is useful for ONEAudit when the original upper bounds were fine but ONEAudit added the contest
            to some CVRs in some pool batches.
        '''
        for c, con in contests.items():
            found = np.sum([cvr.has_contest(c) for cvr in cvrs])
            if found > con.cards:
                if not force:
                    raise ValueError(f'{found} cards contain contest {c} but upper bound is {con.cards}')
                else:
                    warnings.warn(f'{found} cards contain contest {c} but upper bound is {con.cards}')
            con.cards = max(con.cards, found) if force else con.cards

    @classmethod
    def tally(cls, con_dict: dict = None, cvr_list: Collection[CVR] = None, enforce_rules: bool = True) -> dict:
        """
        Tally the votes in the contests in con_dict from a collection of CVRs.
        Only tallies plurality, multi-winner plurality, supermajority, and approval contests.

        Parameters
        ----------
        con_dict: dict
            dict of Contest objects to find tallies for
        cvr_list: list[CVR]
            list of CVRs containing the votes to tally
        enforce_rules: bool
            Enforce the voting rules for the social choice function?
            For instance, if the contest is a vote-for-k plurality and the CVR has more than k votes,
            then if `enforce_rules`, no candidate's total is incremented, but if `not enforce_rules`,
            the tally for every candidate with a vote is incremented.

        Returns
        -------

        Side Effects
        ------------
        Sets the `tally` dict for the contests in con_list, if their social choice function is appropriate
        """
        tallies = {}
        cons = []
        for id, c in con_dict.items():
            if type(c.choice_function) in [Plurality, Threshold, Approval]:
                cons.append(c)
                c.tally = defaultdict(int)
            else:
                warnings.warn(
                    f"contest {c.id} ({c.name}) has social choice function "
                    + f"{c.choice_function}: not tabulated"
                )
        for cvr in cvr_list:
            for c in cons:
                if cvr.has_contest(c.id):
                    if enforce_rules:
                        n_votes = 0
                        for candidate, vote in cvr.votes[c.id].items():
                            if candidate:
                                n_votes += int(bool(vote))
                    if (not enforce_rules) or (n_votes <= c.n_winners):
                        for candidate, vote in cvr.votes[c.id].items():
                            if candidate:
                                c.tally[candidate] += int(bool(vote))

    @classmethod
    def from_dict(cls, d: dict) -> dict:
        """
        define a contest objects from a dict containing data for one contest
        """
        c = Contest()
        c.__dict__.update(d)
        return c

    @classmethod
    def from_dict_of_dicts(cls, d: dict) -> dict:
        """
        define a dict of contest objects from a dict of dicts, each inner dict containing data for one contest
        """
        contests = {}
        for di, v in d.items():
            contests[di] = cls.from_dict(v)
            contests[di].id = di
        return contests

    @classmethod
    def from_cvr_list(
            cls, audit, votes, cards, cvr_list: Collection[CVR] = None
    ) -> dict:
        """
        Create a contest dict containing all contests in cvr_list.
        Every contest is single-winner plurality by default, audited by ballot comparison
        """
        if len(audit.strata) > 1:
            raise NotImplementedError("stratified audits not implemented")
        stratum = next(iter(audit.strata.values()))
        use_style = stratum.use_style
        max_cards = stratum.max_cards
        contest_dict = {}
        for key in votes:
            contest_name = str(key)
            cards_with_contest = cards[key]
            options = np.array(list(votes[key].keys()), dtype="str")
            tallies = np.array(list(votes[key].values()))

            reported_winner = options[np.argmax(tallies)]

            contest_dict[contest_name] = {
                "name": contest_name,
                "cards": cards_with_contest if use_style else max_cards,
                "choice_function": Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                "n_winners": 1,
                "risk_limit": 0.05,
                "candidates": list(options),
                "winner": [reported_winner],
                "assertion_file": None,
                "audit_type": Audit.AUDIT_TYPE.CARD_COMPARISON,
                "test": NonnegMean.alpha_mart,
                "estim": NonnegMean.optimal_comparison,
                "bet": NonnegMean.fixed_bet,
            }
        contests = Contest.from_dict_of_dicts(contest_dict)
        return contests

    @classmethod
    def print_margins(cls, contests: dict = None):
        """
        print all assorter margins
        """
        for c, con in contests.items():
            print(f"margins in contest {c}:")
            for a, m in con.margins.items():
                print(f"\tassertion {a}: {m}")
