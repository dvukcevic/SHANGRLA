import json
import math
import warnings
from collections.abc import Collection
from typing import Type, Callable
import numpy as np
from itertools import permutations

from shangrla.core.NonnegMean import NonnegMean
from shangrla.core.Audit import Stratum, NpEncoder, Audit
from shangrla.core.contest import CVR, Contest
import shangrla.core.contest as scc
from shangrla.core.eprocess import NodeRegistry
import shangrla.core.eprocess as eproc


class Assertion:
    """
    Objects and methods for SHANGRLA assertions about election outcomes

    An _assertion_ is a statement of the form
      "the average value of this assorter applied to the ballots is greater than 1/2"
    An _assorter_ maps votes to nonnegative numbers not exceeding some upper bound, `upper_bound`
    """

    def __init__(
        self,
        contest: Contest = None,
        assorter: callable = None,
        winner: str = None,
        loser: str = None,
        margin: float = None,
        test: NonnegMean = None,
        estim: callable = None,
        bet: callable = None,
        test_kwargs: dict = None,
        node: eproc.Node = None,
        # p_value: float = 1,
        # p_history: list = [],
        proved: bool = False,
        sample_size: int = None,
        tally_pool_means: dict = None,
    ):
        """
        test is an instance of NonnegMean

        Parameters
        ----------
        contest: Contest instance
            contest to which the assorter is relevant
        winner: str
            identifier for the nominal "winner" for this assertion. Can be an element of self.contest.candidates,
            an element of Contest.CANDIDATES, or an arbitrary label.
            Using an element of self.contest.candidates or an element of Contest.CANDIDATES can be useful for
            setting the margin in approval, plurality, and supermajority contests.
        loser: str
            identifier for the nominal "loser" for this assertion. Can be an element of self.contest.candidates,
            an element of Contest.CANDIDATES, or an arbitrary label.
            Using an element of self.contest.candidates or an element of Contest.CANDIDATES can be useful for
            setting the margin in approval, plurality, and supermajority contests.
        assorter: Assorter instance
            the assorter for the assertion
        margin: float
            the assorter margin. Generally this will not be known when the assertion is created, but will be set
            later.
        test: instance of class NonnegMean
            the function to find the p-value of the hypothesis that the assertion is true, i.e., that the
            assorter mean is <=1/2
        node: Node
            contains the current p-value for the complementary null hypothesis that the assertion is false along with
            the history of p-values, sample by sample.
            Generally, using history is valid only for sequential risk-measuring functions.
        proved: boolean
            has the complementary null hypothesis been rejected?
        sample_size: int
            estimated total sample size to complete the audit of this assertion
        tally_pool_means: dict
            dict of reported assorter means for each `tally_pool`, for ONEAudit

        """
        self.contest = contest
        self.winner = winner
        self.loser = loser
        self.assorter = assorter
        self.margin = margin
        self.test: NonnegMean = test
        self.estim = estim
        self.bet = bet
        self.test_kwargs = {} if test_kwargs is None else test_kwargs
        self.node = node
        self.proved = proved
        self.sample_size = sample_size
        self.tally_pool_means = tally_pool_means
        if assorter:
            self.assorter.tally_pool_means = tally_pool_means

    def __str__(self):
        return (
            f"contest_id: {self.contest.id} winner: {self.winner} loser: {self.loser} "
            f"assorter: {str(self.assorter)} p-value: {self.p_value()} "
            f"margin: {self.margin} test: {str(self.test)} estim: {str(self.estim)} bet: {str(self.bet)} "
            f"test_kwargs: {str(self.test_kwargs)} "
            f"p-history length: {len(self.node)} proved: {self.proved} sample_size: {self.sample_size} "
            f"assorter upper bound: {self.assorter.upper_bound} "
            f"proved:  {self.proved} "
            f"sample_size: {self.sample_size} "
        )

    def to_dict(self):
        '''
        Support serialization of the minimal set of relevant ivars; this prevents circular reference when
        attempting to serialize all of self._dict__ (since the Assertion class contains a Contest reference
        and a Contest contains a dict of Assertions)
        '''
        return {
            "contest": self.contest.id,
            "winner": self.winner,
            "loser": self.loser,
            "p_value": self.p_value(),
            "min_p": self.min_p(),
            "margin": self.margin,
            "test": self.test,
            "estim": self.estim,
            "bet": self.bet,
            "test_kwargs": self.test_kwargs,
            "history_length": len(self.node),
            "proved": self.proved,
            "sample_size": self.sample_size,
            "assorter_upper_bound": self.assorter.upper_bound,
        }

    def p_value(self):
        if self.node is None or len(self.node) == 0:
            return 1.0
        return self.node.get_p_value()

    def min_p(self):
        if self.node is None or len(self.node) == 0:
            return 1.0
        return self.node.get_min_p_value()

    def compute_margin(self, cvr_list: Collection = None, use_style: bool = True):
        """
        find the margin for a list of Cvrs.
        By definition, the margin is twice the mean of the assorter, minus 1.

        Parameters
        ----------
        cvr_list: Collection
            collection of cast-vote records

        Returns
        ----------
        margin: float
        """
        return 2 * self.assorter.mean(cvr_list, use_style=use_style) - 1

    def overstatement_assorter_margin(
        self, error_rate_1: float = 0, error_rate_2: float = 0
    ) -> float:
        """
        find the overstatement assorter margin corresponding to an assumed rate of 1-vote and 2-vote overstatements

        Parameters
        ----------
        error_rate_1: float
            the assumed rate of one-vote overstatement errors in the CVRs
        error_rate_2: float
            the assumed rate of two-vote overstatement errors in the CVRs

        Returns
        -------
        the overstatement assorter margin implied by the reported margin and the assumed rates of overstatements
        """
        return (
            1
            - (error_rate_2 + error_rate_1 / 2)
            * self.assorter.upper_bound
            / self.margin
        ) / (2 * self.assorter.upper_bound / self.margin - 1)

    def overstatement_assorter_mean(
        self, error_rate_1: float = 0, error_rate_2: float = 0
    ) -> float:
        """
        find the overstatement assorter mean corresponding to assumed rates of 1-vote and 2-vote overstatements

        Parameters
        ----------
        error_rate_1: float
            the assumed rate of one-vote overstatement errors in the CVRs
        error_rate_2: float
            the assumed rate of two-vote overstatement errors in the CVRs


        Returns
        -------
        overstatement assorter mean implied by the assorter mean and the assumed error rates
        """
        return (1 - error_rate_1 / 2 - error_rate_2) / (
            2 - self.margin / self.assorter.upper_bound
        )

    def overstatement_assorter(
        self, mvr: CVR = None, cvr: CVR = None, use_style=True
    ) -> float:
        """
        assorter that corresponds to normalized overstatement error for an assertion

        If `use_style == True`, then if the CVR contains the contest but the MVR does not,
        that is considered to be an overstatement, because the ballot is presumed to contain
        the contest.

        If `use_style == False`, then if the CVR contains the contest but the MVR does not,
        the MVR is considered to be a non-vote in the contest.

        Parameters
        -----------
        mvr: CVR
            the manual interpretation of voter intent
        cvr: CVR
            the machine-reported cast vote record.

        Returns
        --------
        over: float
            (1-o/u)/(2-v/u), where
                o is the overstatement
                u is the upper bound on the value the assorter assigns to any ballot
                v is the assorter margin
        """
        return (
            1
            - self.assorter.overstatement(mvr, cvr, use_style)
            / self.assorter.upper_bound
        ) / (2 - self.margin / self.assorter.upper_bound)

    def set_margin_from_cvrs(self, audit: Audit = None, cvr_list: Collection = None):
        """
        find assorter margin from cvrs and store it

        Parameters
        ----------
        cvr_list: Collection
            cvrs from which the sample will be drawn
        use_style: bool
            is the sample drawn only from ballots that should contain the contest?

        Returns
        -------
        nothing

        Side effects
        ------------
        sets assorter.margin
        """
        if len(audit.strata) > 1:
            raise NotImplementedError("stratified audits not yet supported")
        stratum = next(iter(audit.strata.values()))
        use_style = stratum.use_style
        amean = self.assorter.mean(cvr_list, use_style=use_style)
        if amean < 1 / 2:
            warnings.warn(
                f"assertion {self} not satisfied by CVRs: mean value is {amean}"
            )
        self.margin = 2 * amean - 1
        if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING:
            self.test.u = self.assorter.upper_bound
        elif self.contest.audit_type in [
            Audit.AUDIT_TYPE.CARD_COMPARISON,
            Audit.AUDIT_TYPE.ONEAUDIT,
        ]:
            self.test.u = 2 / (2 - self.margin / self.assorter.upper_bound)
        else:
            raise NotImplementedError(
                f"audit type {self.contest.audit_type} not supported"
            )
        return self.margin

    def find_margin_from_tally(self, tally: dict = None):
        """
        find the assorter margin between implied by a tally.

        Generally useful only for approval, plurality, and supermajority contests.

        Assumes the number of cards containing the contest has been set.

        Parameters
        ----------
        tally: dict
            dict of tallies for the candidates in the contest. Keys are candidates as listed
            in Contest.candidates. If `tally is None` tries to use the contest.tally.

        The margin for a supermajority contest with a winner is (see SHANGRLA section 2.3)
              2(pq/(2f) + (1 − q)/2 - 1/2) = q(p/f-1), where:
                     q is the fraction of cards that have valid votes
                     p is the fraction of cards that have votes for the winner
                     f is the fraction of valid votes required to win.

        Returns
        -------
        nothing

        Side effects
        ------------
        sets self.margin

        """
        tally = tally if tally else self.contest.tally
        if (
            self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY
            or self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.APPROVAL
        ):
            self.margin = (tally[self.winner] - tally[self.loser]) / self.contest.cards
        elif (
            self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY
        ):
            if (
                self.winner == Contest.CANDIDATES.NO_CANDIDATE
                or self.loser != Contest.CANDIDATES.ALL_OTHERS
            ):
                raise NotImplementedError(
                    f"TO DO: currently only support super-majority with a winner"
                )
            else:
                q = (
                    np.sum([tally[c] for c in self.contest.candidates])
                    / self.contest.cards
                )
                p = tally[self.winner] / self.contest.cards
                self.margin = q * (p / self.contest.share_to_win - 1)
        else:
            raise NotImplementedError(
                f"social choice function {self.contest.choice_function} not supported"
            )

    def make_overstatement(self, overs: float, use_style: bool = False) -> float:
        """
        return the numerical value corresponding to an overstatement of `overs` times the assorter upper bound `u`

        **Assumes that the margin has been set.**

        Parameters
        ----------
        overs: float
            the multiple of `u`
        use_style: bool
            flag to use style information. Only used if the assorter margin has not been set

        Returns
        -------
        the numerical value corresponding to an overstatement of that multiple

        """
        return (1 - overs / self.assorter.upper_bound) / (
            2 - self.margin / self.assorter.upper_bound
        )

    def mvrs_to_data(
        self, mvr_sample: list = None, cvr_sample: list = None, use_all: bool=False
    ) -> np.array:
        """
        Process mvrs (and, for comparison audits, cvrs) to create data for the assertion's test
        and for sample size simulations.

        Creates assorter values for the mvrs, or overstatement assorter values using the mvrs and cvrs,
        according to whether the audit uses ballot polling or card-level comparison

        The margin should be set before calling this function.

        mvr_sample and cvr_sample should be ordered using CVR.prep_comparison_sample() or
           CVR.prep_polling_sample() before calling this routine

        Parameters
        ----------
        mvr_sample: list of CVR objects
            corresponding MVRs
        cvr_sample: list of CVR objects
            sampled CVRs
        use_all: bool
            if True, ignore contest sample_num in determining which pairs to include

        Returns
        -------
        d: np.array
            either assorter values or overstatement assorter values, depending on the audit method
        u: upper bound for the test
        """
        margin = self.margin
        upper_bound = self.assorter.upper_bound
        con = self.contest
        use_style = con.use_style
        if con.audit_type in [
            Audit.AUDIT_TYPE.CARD_COMPARISON,
            Audit.AUDIT_TYPE.ONEAUDIT,
        ]:
            d = np.array(
                [
                    self.overstatement_assorter(
                        mvr_sample[i], cvr_sample[i], use_style=use_style
                    )
                    for i in range(len(mvr_sample))
                    if (
                        (not use_style)
                        or (
                            cvr_sample[i].has_contest(con.id)
                            and (use_all or (cvr_sample[i].sample_num <= con.sample_threshold))
                        )
                    )
                ]
            )
            u = 2 / (2 - margin / upper_bound)
        elif (
            con.audit_type == Audit.AUDIT_TYPE.POLLING
        ):  # Assume style information is irrelevant
            d = np.array(
                [self.assorter.assort(mvr_sample[i]) for i in range(len(mvr_sample))]
            )
            u = upper_bound
        else:
            raise NotImplementedError(f"audit type {con.audit_type} not implemented")
        return d, u

    def find_sample_size(
        self,
        data: np.array = None,
        prefix: bool = False,
        rate_1: float = None,
        rate_2: float = None,
        reps: int = None,
        quantile: float = 0.5,
        seed: int = 1234567890,
    ) -> int:
        """
        Estimate sample size needed to reject the null hypothesis that the assorter mean is <=1/2,
        for the specified risk function, given:
            - for comparison audits, the assorter margin and assumptions about the rate of overstatement errors
            - for polling audits, either a set of assorter values, or the assumption that the reported tallies
              are correct

        If `data is not None`, uses data to make the estimate. There are three strategies:
            1. if `reps is None`, tile the data to make a list of length N
            2. if `reps is not None and not prefix`, sample from the data with replacement to make `reps` lists of
               length N
            3. if `reps is not None and prefix`, start with `data`, then draw N-len(data) times from data with
               replacement to make `reps` lists of length N

        If `data is None`, constructs values from scratch.
            - For polling audits, values are inferred from the reported tallies. Since contest.tally only reports
              actual candidate totals, not IRV/RAIRE pseudo-candidates, this is not implemented for IRV.
            - For comparison audits, there are two strategies to construct the values:
                1. Systematically interleave small and large values, starting with a small value (`reps is None`)
                2. Sample randomly from a set of such values
            The rate of small values is `rate_1` if `rate_1 is not None`. If `rate is None`, for POLLING audits, gets
            the rate of small values from the margin.
            For Audit.AUDIT_TYPE.POLLING audits, the small values are 0 and the large values are `u`; the rest are 1/2.
            For Audit.AUDIT_TYPE.CARD_COMPARISON audits, the small values are the overstatement assorter for an
            overstatement of `u/2` and the large values are the overstatement assorter for an overstatement of 0.

        This function is for a single assertion.

        **Assumes that self.test.u has been set appropriately for the audit type (polling or comparison).**
        **Thus, for comparison audits, the assorter margin should be be set before calling this function.**

        Parameters
        ----------
        data: np.array
            observations on which to base the calculation. If `data is not None`, uses them in a bootstrap
            approach, rather than simulating errors.
            If `self.contest.audit_type==Audit.POLLING`, the data should be (simulated or actual) values of
            the raw assorter.
            If `self.contest.audit_type==Audit.CARD_COMPARISON`, the data should be (simulated or actual)
            values of the overstatement assorter.
        prefix: bool
            prefix the data, then sample or tile to produce the remaining values
        rate_1: float
            assumed rate of "small" values for simulations (1-vote overstatements). Ignored if `data is not None`
            If `rate_1 is None and self.contest.audit_type==Audit.POLLING` the rate of small values is inferred
            from the margin
        rate_2: float
            assumed rate of 0s for simulations (2-vote overstatements).
        reps: int
            if `reps is None`, builds the data systematically
            if `reps is not None`, performs `reps` simulations to estimate the `quantile` quantile of sample size.
        quantile: float
            if `reps is not None`, quantile of the distribution of sample sizes to return
            if `reps is None`, ignored
        seed: int
            if `reps is not None`, use `seed` as the seed in numpy.random to estimate the quantile

        Returns
        -------
        sample_size: int
            sample size estimated to be sufficient to confirm the outcome if data are generated according to
            the assumptions

        Side effects
        ------------
        sets the sample_size attribute of the assertion

        """
        assert self.margin > 0, f"Margin {self.margin} is nonpositive"
        if data is not None:
            sample_size = self.test.sample_size(
                data,
                alpha=self.contest.risk_limit,
                reps=reps,
                prefix=prefix,
                quantile=quantile,
                seed=seed,
            )
        else:
            """
            Construct data.
            For POLLING, values are 0, 1/2, and u.
            For CARD_COMPARISON, values are overstatement assorter values corresponding to
            overstatements of 2u (at rate_2), u (at rate_1), or 0.
            """
            big = (
                self.assorter.upper_bound
                if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING
                else self.make_overstatement(overs=0)
            )
            small = (
                0
                if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING
                else self.make_overstatement(overs=1/2)
            )
            rate_1 = (
                rate_1 if rate_1 is not None else (1 - self.margin) / 2
            )  # rate of small values
            x = big * np.ones(self.test.N)
            if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING:
                if self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.IRV:
                    raise NotImplementedError(
                        f"data must be provided to estimate sample sizes for IRV assertions"
                    )
                else:  # get tally
                    if self.contest.tally:
                        n_0 = self.contest.tally[self.loser]
                        n_big = self.contest.tally[self.winner]
                        n_half = self.test.N - n_0 - n_big
                        x = self.interleave_values(n_0, n_half, n_big, big=big)
                    else:
                        raise ValueError(
                            f"contest {self.contest} tally required but not defined"
                        )
            elif (
                self.contest.audit_type in [Audit.AUDIT_TYPE.CARD_COMPARISON, Audit.AUDIT_TYPE.ONEAUDIT]
            ):  # comparison audit
                rate_1_i = (
                    np.arange(0, self.test.N, step=int(1 / rate_1), dtype=int)
                    if rate_1
                    else []
                )
                rate_2_i = (
                    np.arange(0, self.test.N, step=int(1 / rate_2), dtype=int)
                    if rate_2
                    else []
                )
                x[rate_1_i] = small
                x[rate_2_i] = 0
            else:
                raise NotImplementedError(
                    f"audit type {self.contest.audit_type} for contest {self.contest} not implemented"
                )
            sample_size = self.test.sample_size(
                x,
                alpha=self.contest.risk_limit,
                reps=reps,
                prefix=prefix,
                quantile=quantile,
                seed=seed,
            )
        self.sample_size = sample_size
        return sample_size

    def set_p_values(self, mvr_sample: list[CVR], cvr_sample: list[CVR] = None, use_all=False
    ) -> float:
        """
        Update this assertion's log-evidence history from the supplied sample.

        Parameters
        ----------
        mvr_sample: list of CVR objects
            the manually ascertained voter intent from sheets, including entries for phantoms

        cvr_sample: list of CVR objects
            the cvrs for the same sheets, for ballot-level comparison audits
            not needed for polling audits

        Returns
        -------
        p_value: float
            current p-value for this assertion

        Side effect
        -----------
        Replaces the node's log-evidence history and latches ``proved`` once its
        minimum p-value reaches the contest risk limit.

        """
        if cvr_sample is not None and len(mvr_sample) != len(cvr_sample):
            raise ValueError("unequal numbers of cvrs and mvrs")
        if self.node is None:
            raise ValueError("assertion must have a node before setting p-values")
        d, u = self.mvrs_to_data(mvr_sample, cvr_sample, use_all=use_all)
        self.test.u = u  # set upper bound for the test for each assorter

        _, e_history = self.test.test(d)
        log_e_history = np.log(np.asarray(e_history, dtype=float))
        self.node.eprocess.append_at(log_e_history, 0)
        if self.min_p() <= self.contest.risk_limit:
            self.proved = True
        return self.p_value()

    @classmethod
    def interleave_values(
        cls,
        n_small: int,
        n_med: int,
        n_big: int,
        small: float = 0,
        med: float = 1/2,
        big: float = 1,
    ):
        r"""
        make an interleaved population of n_s values equal to small, n_m values equal to med, and n_big equal to big
        Start with a small if n_small > 0
        """
        N = n_small + n_med + n_big
        x = np.zeros(N)
        i_small = 0
        i_med = 0
        i_big = 0
        r_small = 1 if n_small else 0
        r_med = 1 if n_med else 0
        r_big = 1
        if r_small:  # start with small
            x[0] = small
            i_small = 1
            r_small = (n_small - i_small) / n_small
        elif r_med:  # start with 1/2
            x[0] = med
            i_med = 1
            r_med = (n_med - i_med) / n_med
        else:
            x[0] = big
            i_big = 1
            r_big = (n_big - i_big) / n_big
        for i in range(1, N):
            if r_small > r_big:
                if r_med > r_small:
                    x[i] = med
                    i_med += 1
                    r_med = (n_med - i_med) / n_med
                else:
                    x[i] = small
                    i_small += 1
                    r_small = (n_small - i_small) / n_small
            elif r_med > r_big:
                x[i] = med
                i_med += 1
                r_med = (n_med - i_med) / n_med
            else:
                x[i] = big
                i_big += 1
                r_big = (n_big - i_big) / n_big
        return x

    @classmethod
    def reset_p_values(cls, contests: dict) -> bool:
        '''
        Resets assertion and composition-tree evidence for every contest.
        '''
        for con in contests.values():
            handler = con.assertions
            if isinstance(handler, AssertionHandler):
                handler.reset_p_values()
            else:
                for asn in handler.values():
                    asn.proved = False
                    if asn.node is not None:
                        asn.node.eprocess.reset()
                con.p_values = {a: 1 for a in handler}
                con.proved = {a: False for a in handler}
                con.max_p = 1
        return True


class AssertionHandler:
    # supported json assertion types for imported assertions
    JSON_ASSERTION_TYPES = (
        WINNER_ONLY := "WINNER_ONLY",
        IRV_ELIMINATION := "IRV_ELIMINATION",
    )

    def __init__(self, contest: Contest, **kwargs):
        """
        Construct assertions that imply the winner(s) got more votes than the loser(s).

        The assertions are that every winner beat every loser: there are
        len(winner)*len(loser) pairwise assertions in all.

        Parameters
        -----------
        contest: instance of Contest
            contest to which the assertions are relevant
        winner: list
            list of identifiers of winning candidate(s)
        loser: list
            list of identifiers of losing candidate(s)
        test: NonnegMean
            statistical test to use
        test_kwargs: dict
            kwargs for the test
        estim: callable
            estimator for alpha test supermartingale
        bet: callable
            bet for betting test supermartingale

        Returns
        --------
        a dict of Assertions

        """
        self.contest = contest
        self.scf = contest.choice_function
        self.candidates = contest.candidates
        self.ub = self.scf.assort_upper_bound()
        self.winners = contest.winner
        self.losers = list(set(contest.candidates) - set(self.winners))
        self.test = contest.test
        self.test_kwargs = contest.test_kwargs
        self.estim = contest.estim
        self.bet = contest.bet
        self.params = self.validate_params(**kwargs)

        self.registry: NodeRegistry = NodeRegistry()
        self.assertions: dict[str, Assertion] = {}

        #self.root = self.registry.get_composite_node("root")
        # self.status =
        self.proved: bool = False
        self.stratified: bool = False  # todo add stratum object here

        self.build_assertion_handler()

    def make_assertion(self, node, winner, loser, winner_cands, loser_cands, id=None):
        _test = NonnegMean(
            test=self.test,
            estim=self.estim,
            bet=self.bet,
            g=self.contest.g,
            u=self.ub,
            N=self.contest.cards,
            t=1 / 2,
            random_order=True,
            **self.test_kwargs)
        _assorter = Assorter(
            contest=self.contest,
            assort=self.contest.make_assort_function(winner=winner, loser=loser, winner_cands=winner_cands, loser_cands=loser_cands),
            upper_bound=self.ub)
        assertion = Assertion(
            self.contest,
            winner=winner,
            loser=loser,
            node=node,
            assorter=_assorter,
            test=_test,
            estim=self.estim,
            bet=self.bet,
            test_kwargs=self.test_kwargs
        )
        key_error_extra = ""
        if id is None:
            id = f"{winner} v {loser}"
            key_error_extra = " (default id used)"
        if id in self.assertions:
            raise KeyError(f"Assertion {id} already exists{key_error_extra}")
        self.assertions[id] = assertion

    def build_assertion_handler(self):
        get_assertion_builder(self.scf)(self, self.winners, self.losers)

    def set_all_margins_from_cvrs(  # TODO this should be called by Audit class looping over contests
        self,
        audit: Audit = None,
        # contests: dict = None,
        cvr_list: Collection[CVR] = None,
    ):
        """
        Find all the assorter margins in a set of Assertions. Updates the dict of dicts of assertions
        and the contest dict.

        Appropriate only if cvrs are available. Otherwise, base margins on the reported results.

        This function is primarily about side-effects on the assertions in the contest dict.

        Parameters
        ----------
        audit: Audit
            information about the audit
        contests: dict of Contest objects
        cvr_list: Collection
            collection of CVR objects

        Returns
        -------
        min_margin: float
            smallest margin in the audit

        Side effects
        ------------
        sets the margin of every assertion
        sets the assertion.test.u for every assertion, according to whether
           `assertion.contest.audit_type==Audit.AUDIT_TYPE.POLLING`
           or `assertion.contest.audit_type in [Audit.AUDIT_TYPE.CARD_COMPARISON, Audit.AUDIT_TYPE.ONEAUDIT]`
        """
        min_margin = np.inf
        for a, asrt in self.assertions.items():
            margin = asrt.set_margin_from_cvrs(audit, cvr_list)
            min_margin = np.min([min_margin, margin])
        return min_margin

    def set_p_values(self, mvr_sample: list, cvr_sample: list = None, use_all=False) -> float:
        """
        Update every assertion, the composed contest process, and contest summaries.

        Parameters
        ----------
        mvr_sample: list of CVR objects
            the manually ascertained voter intent from sheets, including entries for phantoms

        cvr_sample: list of CVR objects
            the cvrs for the same sheets, for ballot-level comparison audits
            not needed for polling audits

        Returns
        -------
        max_p: float
            largest current p-value among this contest's assertions

        Side effects
        ------------
        Updates each assertion's node history and proof flag, rebuilds the
        composition tree, and updates ``contest.p_values``, ``contest.proved``,
        and ``contest.max_p``.

        """
        if cvr_sample is not None and len(mvr_sample) != len(cvr_sample):
            raise ValueError("unequal numbers of cvrs and mvrs")
        self.contest.p_values = {}
        self.contest.proved = {}
        p_values = {}
        proved = {}
        for a, asrt in self.assertions.items():
            p_values[a] = asrt.set_p_values(mvr_sample, cvr_sample, use_all=use_all)
            proved[a] = asrt.proved

        self._recompute_composition()
        self.contest.p_values = p_values
        self.contest.proved = proved
        contest_max_p = max(p_values.values(), default=1.0)
        self.contest.max_p = contest_max_p
        root_p = self._node_p_value(self.registry.root)
        if root_p <= self.contest.risk_limit:
            self.proved = True
        return contest_max_p

    @staticmethod
    def _node_p_value(node: eproc.Node) -> float:
        return 1.0 if len(node) == 0 else node.get_p_value()

    def _recompute_composition(self) -> None:
        """Rebuild composed processes from the current assertion leaf histories."""
        visited = set()

        def recompute(node):
            if not isinstance(node, eproc.CompositeNode) or id(node) in visited:
                return
            visited.add(id(node))
            for child in node.children:
                recompute(child)
            node.eprocess.reset()
            node.combiner.reset()
            if node.children:
                end = min(len(child) for child in node.children)
                if end:
                    node.combine(start=0, end=end)

        recompute(self.registry.root)

    def reset_p_values(self) -> bool:
        """Reset assertion evidence, composed histories, and proof status."""
        for node in self.registry.nodes.values():
            node.eprocess.reset()
            if isinstance(node, eproc.CompositeNode):
                node.combiner.reset()
        self.registry.root.eprocess.reset()
        if isinstance(self.registry.root, eproc.CompositeNode):
            self.registry.root.combiner.reset()

        self.contest.p_values = {}
        self.contest.proved = {}
        for assertion_id, assertion in self.assertions.items():
            assertion.proved = False
            self.contest.p_values[assertion_id] = 1.0
            self.contest.proved[assertion_id] = False
        self.contest.max_p = 1.0
        self.proved = False
        return True


    @staticmethod
    def validate_params(**kwargs) -> dict:
        params = {}

        # TODO list and define all possible params here. Throw error or warning if unexpected params provided

        return kwargs


def make_all_assertions(contests: dict[str, Contest]) -> None:
    """
    Construct all the assertions to audit the contests and add the assertions to the contest dict

    Parameters
    ----------
    contests: dict
        dict of Contest objects

    Returns
    -------
    True

    Side Effects
    ------------
    creates assertions and adds the dict of assertions relevant to each contest to the contest
    object's `assertions` attribute

    """
    for c, con in contests.items():
        contests[c].assertions = AssertionHandler(contest=con)


class Assorter:
    """
    Class for generic Assorter.

    An assorter must either have an `assort` method or both `winner` and `loser` must be defined
    (in which case assort(c) = (winner(c) - loser(c) + 1)/2. )

    Class parameters:
    -----------------
    contest: Contest instance
        the contest to which this Assorter applies

    winner: callable
        maps a dict of selections into the value 1 if the dict represents a vote for the winner

    loser : callable
        maps a dict of selections into the value 1 if the dict represents a vote for the winner

    assort: callable
        maps dict of selections into float

    upper_bound: float
        a priori upper bound on the value the assorter assigns to any dict of selections

    tally_pool_means: dict
        mean of the assorter over each tally_pool of CVRs, for ONEAudit

    The basic method is assort, but the constructor can be called with (winner, loser)
    instead. In that case,

        assort = (winner - loser + 1)/2

    """

    def __init__(
        self,
        contest: Contest = None,
        assort: callable = None,
        winner: str = None,
        loser: str = None,
        upper_bound: float = 1,
        tally_pool_means: dict = None,
    ):
        """
        Constructs an Assorter.

        If assort is defined and callable, it becomes the class instance of assort

        If assort is None but both winner and loser are defined and callable,
           assort is defined to be 1/2 if winner=loser; winner, otherwise


        Parameters
        -----------
        con: Contest instance
            the contest to which the assorter is relevant
        assort: callable
            maps a dict of votes into [0, upper_bound]
        winner: callable
            maps a pattern into [0, 1]
        loser: callable
            maps a pattern into [0, 1]
        upper_bound: float > 0
            a priori upper bound on the value the assorter can take
        tally_pool_means: dict
            dict of the mean value of the assorter over each tally_pool of CVRs

        """
        self.contest = contest
        self.winner = winner
        self.loser = loser
        self.upper_bound = upper_bound
        if assort is not None:
            assert callable(assort), "assort must be callable"
            # self.assort = assort
            def f(x):
                # print(x, assort(x), contest)
                return assort(x)
            self.assort = f
        else:
            assert callable(winner), "winner must be callable if assort is None"
            assert callable(loser), "loser must be callable if assort is None"
            self.assort = lambda cvr: (self.winner(cvr) - self.loser(cvr) + 1) / 2
        self.tally_pool_means = tally_pool_means

    def __str__(self):
        """
        string representation
        """
        return (
            f"contest_id: {self.contest.id}\nupper bound: {self.upper_bound}, "
            + f"winner defined: {callable(self.winner)}, loser defined: {callable(self.loser)}, "
            + f"assort defined: {callable(self.assort)} "
            + f"tally_pool_means: {bool(self.tally_pool_means)}"
        )

    def mean(self, cvr_list: "Collection[CVR]" = None, use_style: bool = True):
        """
        find the mean of the assorter applied to a list of CVRs

        Parameters
        ----------
        cvr_list: Collection
            a collection of cast-vote records
        use_style: Boolean
            does the audit use card style information? If so, apply the assorter only to CVRs
            that contain the contest in question.

        Returns
        -------
        mean: float
            the mean value of the assorter over the collection of cvrs. If use_style, ignores CVRs that
            do not contain the contest.
        """
        if use_style:
            filtr = lambda c: c.has_contest(self.contest.id)
        else:
            filtr = lambda c: True
        return np.mean([self.assort(c) for c in cvr_list if filtr(c)])

    def set_tally_pool_means(
        self,
        cvr_list: Collection['CVR'] = None,
        tally_pools: Collection = None,
        use_style: bool = True,
    ):
        """
        create dict of pool means for the assorter from a set of CVRs

        Parameters
        ----------
        cvr_list: Collection
            cvrs from which the sample will be drawn

        tally_pools: Collection [optional]
            the labels of the tally groups

        Returns
        -------
        nothing

        Side effects
        ------------
        sets self.tally_pool_means
        """
        if not tally_pools:
            tally_pools = set(c.tally_pool for c in cvr_list if c.pool)
        tally_pool_dict = {}
        for p in tally_pools:
            tally_pool_dict[p] = {}
            tally_pool_dict[p]["n"] = 0
            tally_pool_dict[p]["tot"] = 0
        if use_style:
            filtr = lambda c: c.has_contest(self.contest.id)
        else:
            filtr = lambda c: True
        for c in [cvr for cvr in cvr_list if (filtr(cvr) and cvr.pool)]:
            tally_pool_dict[c.tally_pool]["n"] += 1
            tally_pool_dict[c.tally_pool]["tot"] += self.assort(c)
        self.tally_pool_means = {}
        for p in tally_pools:
            self.tally_pool_means[p] = (
                np.nan
                if tally_pool_dict[p]["n"] == 0
                else tally_pool_dict[p]["tot"] / tally_pool_dict[p]["n"]
            )

    def sum(self, cvr_list: Collection['CVR'] = None, use_style: bool = True):
        """
        find the sum of the assorter applied to a list of CVRs

        Parameters
        ----------
        cvr_list: Collection of CVRs
            a collection of cast-vote records
        use_style: Boolean
            does the audit use card style information? If so, apply the assorter only to CVRs
            that contain the contest in question.

        Returns
        -------
        sum: float
            sum of the value of the assorter over a list of CVRs. If use_style, ignores CVRs that
            do not contain the contest.
        """
        if use_style:
            filtr = lambda c: c.has_contest(self.contest.id)
        else:
            filtr = lambda c: True
        return np.sum([self.assort(c) for c in cvr_list if filtr(c)])

    def overstatement(self, mvr, cvr, use_style=True):
        """
        overstatement error for a CVR compared to the human reading of the ballot

        If use_style, then if the CVR contains the contest but the MVR does
        not, treat the MVR as having a vote for the loser (assort()=0)

        If not use_style, then if the CVR contains the contest but the MVR does not,
        the MVR is considered to be a non-vote in the contest (assort()=1/2).

        Phantom CVRs and MVRs are treated specially:
            A phantom CVR is considered a non-vote in every contest (assort()=1/2).
            A phantom MVR is considered a vote for the loser (i.e., assort()=0) in every
            contest.

        Parameters
        ----------
        mvr: Cvr
            the manual interpretation of voter intent
        cvr: Cvr
            the machine-reported cast vote record

        Returns
        -------
        overstatement: float
            the overstatement error
        """
        # sanity check
        if use_style and not cvr.has_contest(self.contest.id):
            raise ValueError(
                f"use_style==True but {cvr=} does not contain contest {self.contest.id}"
            )
        # assort the MVR
        mvr_assort = (
            0
            if
                mvr.phantom or (use_style and not mvr.has_contest(self.contest.id))
            else
                self.assort(mvr)
        )
        # assort the CVR
        cvr_assort = (
            self.tally_pool_means[cvr.tally_pool]
            if
                cvr.pool and self.tally_pool_means is not None
            else
                int(cvr.phantom) / 2 + (1 - int(cvr.phantom)) * self.assort(cvr)
        )
        return cvr_assort - mvr_assort


type AssertionBuilder = Callable[[AssertionHandler, list[str], list[str]], None]
"""
Type alias for assertion builder functions.

These functions take an AssertionHandler instance, a lists of winners, and a list of losers,
and return None. The assertion handler is updated via side-effects to reflect the 

The function signature is as follows:
def assertion_builder(
    handler: AssertionHandler,
    winners: list[str],
    losers: list[str]
) -> None:
    pass
"""


_builder_registry: dict[Type[scc.SocialChoiceFunction], AssertionBuilder] = {}


def link_with_scf(scf_cls: Type[scc.SocialChoiceFunction]):
    """Decorator to register an assertion‑builder for a given SCF."""
    def decorator(fn: AssertionBuilder):
        _builder_registry[scf_cls] = fn
        return fn
    return decorator


def get_assertion_builder(scf: scc.SocialChoiceFunction) -> AssertionBuilder:
    builder_fn = _builder_registry.get(type(scf))
    if not builder_fn:
        raise NotImplementedError(
            f"Social choice function {scf} is not implemented: No assertion builder registered for {type(scf).__name__}"
        )
    return builder_fn


@link_with_scf(scc.Plurality)
def plurality_builder(handler: AssertionHandler, winners: list[str], losers: list[str]) -> None:
    reg = NodeRegistry()
    root = reg.get_composite_node("root")
    root.combiner = eproc.Minimum()
    reg.set_root(root)
    cands = list(handler.candidates)
    for winner in winners:
        for loser in losers:
            wl_pair = winner + " v " + loser
            node, _ = reg.get_leaf_node(wl_pair)
            root.add_child(node)
            handler.make_assertion(node, winner, loser, cands, cands, id=wl_pair)
    handler.registry = reg


@link_with_scf(scc.Threshold)
def threshold_builder(handler: AssertionHandler, winners: list[str], losers: list[str]) -> None:
    r"""
    TODO UPDATE DOCSTRING
    Construct assertion that winner got >= share_to_win \in (0,1) of the valid votes

    **TO DO: This method assumes there was a winner. To audit that there was no winner requires
    flipping things.**

    An equivalent condition is:

    (votes for winner)/(2*share_to_win) + (invalid votes)/2 > 1/2.

    Thus the correctness of a super-majority outcome--where share_to_win >= 1/2--can
    be checked with a single assertion.

    share_to_win < 1/2 might be useful for some social choice functions, including
    primaries where candidates who receive less than some threshold share are
    eliminated.

    A CVR with a mark for more than one candidate in the contest is considered an
    invalid vote.

    Parameters
    -----------
    contest:
        contest object instance to which the assertion applies
    share_to_win: float
        fraction of the valid votes the winner must get to win
    winner:
        identifier of winning candidate
    loser: list
        list of identifiers of losing candidate(s)
    test: instance of NonnegMean
        risk function for the contest
    test_kwargs: dict
        kwargs for the test
    estim: an estimation method of NonnegMean
        estimator the alpha_mart test uses for the alternative
    bet: method to choose the bet for betting_mart risk function

    Returns
    --------
    a dict containing one Assertion

    """
    reg = NodeRegistry()
    root = reg.get_composite_node("root")
    root.combiner = eproc.Minimum()
    reg.set_root(root)
    loser = Contest.CANDIDATES.ALL_OTHERS
    cands = list(handler.candidates)
    for winner in winners:
        wl_pair = winner + " v " + Contest.CANDIDATES.ALL_OTHERS
        node, _ = reg.get_leaf_node(wl_pair)
        root.add_child(node)
        handler.make_assertion(node, winner, loser, cands, cands, id=wl_pair)
    handler.registry = reg


@link_with_scf(scc.InstantRunoff)
def instant_runoff_builder(handler: AssertionHandler, winners: list[str], losers: list[str]) -> None:
    reg = NodeRegistry()
    root = reg.get_composite_node("root")
    root.combiner = eproc.Minimum()
    reg.set_root(root)
    if 'assertion_json' in dir(handler.contest):  # Check if assertion_json is provided. TODO this is ugly
        # RAIRE style
        for assrtn in handler.contest.assertion_json:
            winner = assrtn["winner"]
            loser = assrtn["loser"]
            if assrtn["assertion_type"] == handler.WINNER_ONLY:
                winner_cands = list(handler.candidates)
                loser_cands = [winner, loser]
                assrtn_id = winner + " v " + loser
            elif assrtn["assertion_type"] == handler.IRV_ELIMINATION:
                # Context is that all candidates in 'eliminated' have been
                # eliminated and their votes distributed to later preferences
                elim = [e for e in assrtn["already_eliminated"]]
                remn = [c for c in handler.candidates if c not in elim]
                winner_cands = remn
                loser_cands = remn
                # Identifier for tracking which assertions have been proved
                assrtn_id = winner + " v " + loser + " elim " + " ".join(elim)
            else:
                raise NotImplemented(
                    f'JSON assertion type {assrtn["assertion_type"]} not implemented.'
                )
            node, _ = reg.get_leaf_node(assrtn_id)
            root.add_child(node)
            handler.make_assertion(node, winner, loser, winner_cands, loser_cands, id=assrtn_id)
    else:
        cands = handler.candidates  # TODO use winners+losers ?
        altorders = permutations(cands)  # representation is [..., 3rd place, runner-up, winner]
        for altorder in altorders:
            if altorder[-1] in winners:  # TODO only supports one winner, throw exception somewhere
                continue  # skip non-altorders
            node = reg.get_composite_node(str(altorder))
            node.combiner = eproc.Linear()
            root.add_child(node)
            for i, newc in enumerate(altorder):
                remaining = frozenset(altorder[i:])
                eliminated = frozenset(altorder) - remaining
                for remc in remaining - {newc}:  # all candidates remaining after newc is eliminated
                    req_name = f"{remc} DB {newc}, {remaining} remaining"
                    requirement, is_new = reg.get_leaf_node(req_name)
                    node.add_child(requirement)
                    if is_new:
                        handler.make_assertion(requirement, remc, newc, remaining, remaining, id=req_name)
                for elimc in eliminated:
                    req_name = f"{elimc} DND {newc}"
                    requirement, is_new = reg.get_leaf_node(req_name)
                    node.add_child(requirement)
                    if is_new:
                        handler.make_assertion(requirement, elimc, newc, cands, [newc, elimc], id=req_name)
                # reqs += [ReqIdent("DND", first=other, second=newc) for other in eliminated]
    handler.registry = reg
