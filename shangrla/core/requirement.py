# from abc import ABC, abstractmethod
# from collections.abc import Collection
# from dataclasses import dataclass
# import numpy as np
# from itertools import permutations
#
# from .Audit import CVR, Contest, Audit
# from .NonnegMean import *
# from .eprocess import NodeRegistry, Node
#
# import shangrla.core.contest as contest
#
#
# class Assertion(ABC):
#     """
#     Objects and methods for SHANGRLA assertions about election outcomes
#
#     An _assertion_ is a statement of the form
#       "the average value of this assorter applied to the ballots is greater than 1/2"
#     An _assorter_ maps votes to nonnegative numbers not exceeding some upper bound, `upper_bound`
#     """
#
#     # supported json assertion types for imported assertions
#     JSON_ASSERTION_TYPES = (
#         WINNER_ONLY := "WINNER_ONLY",
#         IRV_ELIMINATION := "IRV_ELIMINATION",
#     )
#
#     def __init__(
#         self,
#         contest: object = None,
#         assorter: callable = None,
#         winner: str = None,
#         loser: str = None,
#         margin: float = None,
#         test: object = None,
#         estim: callable = None,
#         bet: callable = None,
#         test_kwargs: dict = {},
#         p_value: float = 1,
#         p_history: list = [],
#         proved: bool = False,
#         sample_size: int = None,
#         tally_pool_means: dict = None,
#     ):
#         """
#         test is an instance of NonnegMean
#
#         Parameters
#         ----------
#         contest: Contest instance
#             contest to which the assorter is relevant
#         winner: str
#             identifier for the nominal "winner" for this assertion. Can be an element of self.contest.candidates,
#             an element of Contest.CANDIDATES, or an arbitrary label.
#             Using an element of self.contest.candidates or an element of Contest.CANDIDATES can be useful for
#             setting the margin in approval, plurality, and supermajority contests.
#         loser: str
#             identifier for the nominal "loser" for this assertion. Can be an element of self.contest.candidates,
#             an element of Contest.CANDIDATES, or an arbitrary label.
#             Using an element of self.contest.candidates or an element of Contest.CANDIDATES can be useful for
#             setting the margin in approval, plurality, and supermajority contests.
#         assorter: Assorter instance
#             the assorter for the assertion
#         margin: float
#             the assorter margin. Generally this will not be known when the assertion is created, but will be set
#             later.
#         test: instance of class NonnegMean
#             the function to find the p-value of the hypothesis that the assertion is true, i.e., that the
#             assorter mean is <=1/2
#         p_value: float
#             the current p-value for the complementary null hypothesis that the assertion is false
#         p_history: list
#             the history of p-values, sample by sample. Generally, it is valid only for sequential risk-measuring
#             functions.
#         proved: boolean
#             has the complementary null hypothesis been rejected?
#         sample_size: int
#             estimated total sample size to complete the audit of this assertion
#         tally_pool_means: dict
#             dict of reported assorter means for each `tally_pool`, for ONEAudit
#
#         """
#         self.contest = contest
#         self.winner = winner
#         self.loser = loser
#         self.assorter = assorter
#         self.margin = margin
#         self.test = test
#         self.estim = estim
#         self.bet = bet
#         self.test_kwargs = test_kwargs
#         self.p_value = p_value
#         self.p_history = p_history
#         self.proved = proved
#         self.sample_size = sample_size
#         self.tally_pool_means = tally_pool_means
#         if assorter:
#             self.assorter.tally_pool_means = tally_pool_means
#
#     def __str__(self):
#         return (
#             f"contest_id: {self.contest.id} winner: {self.winner} loser: {self.loser} "
#             f"assorter: {str(self.assorter)} p-value: {self.p_value} "
#             f"margin: {self.margin} test: {str(self.test)} estim: {str(self.estim)} bet: {str(self.bet)} "
#             f"test_kwargs: {str(self.test_kwargs)} "
#             f"p-history length: {len(self.p_history)} proved: {self.proved} sample_size: {self.sample_size} "
#             f"assorter upper bound: {self.assorter.upper_bound} "
#             f"proved:  {self.proved} "
#             f"sample_size: {self.sample_size} "
#         )
#
#     def to_dict(self):
#         '''
#         Support serialization of the minimal set of relevant ivars; this prevents circular reference when
#         attempting to serialize all of self._dict__ (since the Assertion class contains a Contest reference
#         and a Contest contains a dict of Assertions)
#         '''
#         return {
#             "contest": self.contest.id,
#             "winner": self.winner,
#             "loser": self.loser,
#             "p_value": self.p_value,
#             "margin": self.margin,
#             "test": self.test,
#             "estim": self.estim,
#             "bet": self.bet,
#             "test_kwargs": self.test_kwargs,
#             "p_history_length": len(self.p_history),
#             "proved": self.proved,
#             "sample_size": self.sample_size,
#             "assorter_upper_bound": self.assorter.upper_bound,
#         }
#
#     def min_p(self):
#         return min(self.p_history)
#
#     def margin(self, cvr_list: Collection = None, use_style: bool = True):
#         """
#         find the margin for a list of Cvrs.
#         By definition, the margin is twice the mean of the assorter, minus 1.
#
#         Parameters
#         ----------
#         cvr_list: Collection
#             collection of cast-vote records
#
#         Returns
#         ----------
#         margin: float
#         """
#         return 2 * self.assorter.mean(cvr_list, use_style=use_style) - 1
#
#     def overstatement_assorter_margin(
#         self, error_rate_1: float = 0, error_rate_2: float = 0
#     ) -> float:
#         """
#         find the overstatement assorter margin corresponding to an assumed rate of 1-vote and 2-vote overstatements
#
#         Parameters
#         ----------
#         error_rate_1: float
#             the assumed rate of one-vote overstatement errors in the CVRs
#         error_rate_2: float
#             the assumed rate of two-vote overstatement errors in the CVRs
#
#         Returns
#         -------
#         the overstatement assorter margin implied by the reported margin and the assumed rates of overstatements
#         """
#         return (
#             1
#             - (error_rate_2 + error_rate_1 / 2)
#             * self.assorter.upper_bound
#             / self.margin
#         ) / (2 * self.assorter.upper_bound / self.margin - 1)
#
#     def overstatement_assorter_mean(
#         self, error_rate_1: float = 0, error_rate_2: float = 0
#     ) -> float:
#         """
#         find the overstatement assorter mean corresponding to assumed rates of 1-vote and 2-vote overstatements
#
#         Parameters
#         ----------
#         error_rate_1: float
#             the assumed rate of one-vote overstatement errors in the CVRs
#         error_rate_2: float
#             the assumed rate of two-vote overstatement errors in the CVRs
#
#
#         Returns
#         -------
#         overstatement assorter mean implied by the assorter mean and the assumed error rates
#         """
#         return (1 - error_rate_1 / 2 - error_rate_2) / (
#             2 - self.margin / self.assorter.upper_bound
#         )
#
#     def overstatement_assorter(
#         self, mvr: CVR = None, cvr: CVR = None, use_style=True
#     ) -> float:
#         """
#         assorter that corresponds to normalized overstatement error for an assertion
#
#         If `use_style == True`, then if the CVR contains the contest but the MVR does not,
#         that is considered to be an overstatement, because the ballot is presumed to contain
#         the contest.
#
#         If `use_style == False`, then if the CVR contains the contest but the MVR does not,
#         the MVR is considered to be a non-vote in the contest.
#
#         Parameters
#         -----------
#         mvr: CVR
#             the manual interpretation of voter intent
#         cvr: CVR
#             the machine-reported cast vote record.
#
#         Returns
#         --------
#         over: float
#             (1-o/u)/(2-v/u), where
#                 o is the overstatement
#                 u is the upper bound on the value the assorter assigns to any ballot
#                 v is the assorter margin
#         """
#         return (
#             1
#             - self.assorter.overstatement(mvr, cvr, use_style)
#             / self.assorter.upper_bound
#         ) / (2 - self.margin / self.assorter.upper_bound)
#
#     def set_margin_from_cvrs(self, audit: object = None, cvr_list: Collection = None):
#         """
#         find assorter margin from cvrs and store it
#
#         Parameters
#         ----------
#         cvr_list: Collection
#             cvrs from which the sample will be drawn
#         use_style: bool
#             is the sample drawn only from ballots that should contain the contest?
#
#         Returns
#         -------
#         nothing
#
#         Side effects
#         ------------
#         sets assorter.margin
#         """
#         if len(audit.strata) > 1:
#             raise NotImplementedError("stratified audits not yet supported")
#         stratum = next(iter(audit.strata.values()))
#         use_style = stratum.use_style
#         amean = self.assorter.mean(cvr_list, use_style=use_style)
#         if amean < 1 / 2:
#             warnings.warn(
#                 f"assertion {self} not satisfied by CVRs: mean value is {amean}"
#             )
#         self.margin = 2 * amean - 1
#         if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING:
#             self.test.u = self.assorter.upper_bound
#         elif self.contest.audit_type in [
#             Audit.AUDIT_TYPE.CARD_COMPARISON,
#             Audit.AUDIT_TYPE.ONEAUDIT,
#         ]:
#             self.test.u = 2 / (2 - self.margin / self.assorter.upper_bound)
#         else:
#             raise NotImplementedError(
#                 f"audit type {self.contest.audit_type} not supported"
#             )
#
#     @abstractmethod
#     def find_margin_from_tally(self, tally: dict = None):
#         """
#         find the assorter margin between implied by a tally.
#
#         Generally useful only for approval, plurality, and supermajority contests.
#
#         Assumes the number of cards containing the contest has been set.
#
#         Parameters
#         ----------
#         tally: dict
#             dict of tallies for the candidates in the contest. Keys are candidates as listed
#             in Contest.candidates. If `tally is None` tries to use the contest.tally.
#
#         The margin for a supermajority contest with a winner is (see SHANGRLA section 2.3)
#               2(pq/(2f) + (1 − q)/2 - 1/2) = q(p/f-1), where:
#                      q is the fraction of cards that have valid votes
#                      p is the fraction of cards that have votes for the winner
#                      f is the fraction of valid votes required to win.
#
#         Returns
#         -------
#         nothing
#
#         Side effects
#         ------------
#         sets self.margin
#
#         """
#         tally = tally if tally else self.contest.tally
#         if (
#             self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY
#             or self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.APPROVAL
#         ):
#             self.margin = (tally[self.winner] - tally[self.loser]) / self.contest.cards
#         elif (
#             self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY
#         ):
#             if (
#                 self.winner == Contest.CANDIDATES.NO_CANDIDATE
#                 or self.loser != Contest.CANDIDATES.ALL_OTHERS
#             ):
#                 raise NotImplementedError(
#                     f"TODO: currently only support super-majority with a winner"
#                 )
#             else:
#                 q = (
#                     np.sum([tally[c] for c in self.contest.candidates])
#                     / self.contest.cards
#                 )
#                 p = tally[self.winner] / self.contest.cards
#                 self.margin = q * (p / self.contest.share_to_win - 1)
#         else:
#             raise NotImplementedError(
#                 f"social choice function {self.contest.choice_function} not supported"
#             )
#
#     def make_overstatement(self, overs: float, use_style: bool = False) -> float:
#         """
#         return the numerical value corresponding to an overstatement of `overs` times the assorter upper bound `u`
#
#         **Assumes that the margin has been set.**
#
#         Parameters
#         ----------
#         overs: float
#             the multiple of `u`
#         use_style: bool
#             flag to use style information. Only used if the assorter margin has not been set
#
#         Returns
#         -------
#         the numerical value corresponding to an overstatement of that multiple
#
#         """
#         return (1 - overs / self.assorter.upper_bound) / (
#             2 - self.margin / self.assorter.upper_bound
#         )
#
#     def mvrs_to_data(
#         self, mvr_sample: list = None, cvr_sample: list = None, use_all: bool=False
#     ) -> np.array:
#         """
#         Process mvrs (and, for comparison audits, cvrs) to create data for the assertion's test
#         and for sample size simulations.
#
#         Creates assorter values for the mvrs, or overstatement assorter values using the mvrs and cvrs,
#         according to whether the audit uses ballot polling or card-level comparison
#
#         The margin should be set before calling this function.
#
#         mvr_sample and cvr_sample should be ordered using CVR.prep_comparison_sample() or
#            CVR.prep_polling_sample() before calling this routine
#
#         Parameters
#         ----------
#         mvr_sample: list of CVR objects
#             corresponding MVRs
#         cvr_sample: list of CVR objects
#             sampled CVRs
#         use_all: bool
#             if True, ignore contest sample_num in determining which pairs to include
#
#         Returns
#         -------
#         d: np.array
#             either assorter values or overstatement assorter values, depending on the audit method
#         u: upper bound for the test
#         """
#         margin = self.margin
#         upper_bound = self.assorter.upper_bound
#         con = self.contest
#         use_style = con.use_style
#         if con.audit_type in [
#             Audit.AUDIT_TYPE.CARD_COMPARISON,
#             Audit.AUDIT_TYPE.ONEAUDIT,
#         ]:
#             d = np.array(
#                 [
#                     self.overstatement_assorter(
#                         mvr_sample[i], cvr_sample[i], use_style=use_style
#                     )
#                     for i in range(len(mvr_sample))
#                     if (
#                         (not use_style)
#                         or (
#                             cvr_sample[i].has_contest(con.id)
#                             and (use_all or (cvr_sample[i].sample_num <= con.sample_threshold))
#                         )
#                     )
#                 ]
#             )
#             u = 2 / (2 - margin / upper_bound)
#         elif (
#             con.audit_type == Audit.AUDIT_TYPE.POLLING
#         ):  # Assume style information is irrelevant
#             d = np.array(
#                 [self.assorter.assort(mvr_sample[i]) for i in range(len(mvr_sample))]
#             )
#             u = upper_bound
#         else:
#             raise NotImplementedError(f"audit type {con.audit_type} not implemented")
#         return d, u
#
#     @abstractmethod
#     def find_sample_size(
#         self,
#         data: np.array = None,
#         prefix: bool = False,
#         rate_1: float = None,
#         rate_2: float = None,
#         reps: int = None,
#         quantile: float = 0.5,
#         seed: int = 1234567890,
#     ) -> int:
#         """
#         Estimate sample size needed to reject the null hypothesis that the assorter mean is <=1/2,
#         for the specified risk function, given:
#             - for comparison audits, the assorter margin and assumptions about the rate of overstatement errors
#             - for polling audits, either a set of assorter values, or the assumption that the reported tallies
#               are correct
#
#         If `data is not None`, uses data to make the estimate. There are three strategies:
#             1. if `reps is None`, tile the data to make a list of length N
#             2. if `reps is not None and not prefix`, sample from the data with replacement to make `reps` lists of
#                length N
#             3. if `reps is not None and prefix`, start with `data`, then draw N-len(data) times from data with
#                replacement to make `reps` lists of length N
#
#         If `data is None`, constructs values from scratch.
#             - For polling audits, values are inferred from the reported tallies. Since contest.tally only reports
#               actual candidate totals, not IRV/RAIRE pseudo-candidates, this is not implemented for IRV.
#             - For comparison audits, there are two strategies to construct the values:
#                 1. Systematically interleave small and large values, starting with a small value (`reps is None`)
#                 2. Sample randomly from a set of such values
#             The rate of small values is `rate_1` if `rate_1 is not None`. If `rate is None`, for POLLING audits, gets
#             the rate of small values from the margin.
#             For Audit.AUDIT_TYPE.POLLING audits, the small values are 0 and the large values are `u`; the rest are 1/2.
#             For Audit.AUDIT_TYPE.CARD_COMPARISON audits, the small values are the overstatement assorter for an
#             overstatement of `u/2` and the large values are the overstatement assorter for an overstatement of 0.
#
#         This function is for a single assertion.
#
#         **Assumes that self.test.u has been set appropriately for the audit type (polling or comparison).**
#         **Thus, for comparison audits, the assorter margin should be be set before calling this function.**
#
#         Parameters
#         ----------
#         data: np.array
#             observations on which to base the calculation. If `data is not None`, uses them in a bootstrap
#             approach, rather than simulating errors.
#             If `self.contest.audit_type==Audit.POLLING`, the data should be (simulated or actual) values of
#             the raw assorter.
#             If `self.contest.audit_type==Audit.CARD_COMPARISON`, the data should be (simulated or actual)
#             values of the overstatement assorter.
#         prefix: bool
#             prefix the data, then sample or tile to produce the remaining values
#         rate_1: float
#             assumed rate of "small" values for simulations (1-vote overstatements). Ignored if `data is not None`
#             If `rate_1 is None and self.contest.audit_type==Audit.POLLING` the rate of small values is inferred
#             from the margin
#         rate_2: float
#             assumed rate of 0s for simulations (2-vote overstatements).
#         reps: int
#             if `reps is None`, builds the data systematically
#             if `reps is not None`, performs `reps` simulations to estimate the `quantile` quantile of sample size.
#         quantile: float
#             if `reps is not None`, quantile of the distribution of sample sizes to return
#             if `reps is None`, ignored
#         seed: int
#             if `reps is not None`, use `seed` as the seed in numpy.random to estimate the quantile
#
#         Returns
#         -------
#         sample_size: int
#             sample size estimated to be sufficient to confirm the outcome if data are generated according to
#             the assumptions
#
#         Side effects
#         ------------
#         sets the sample_size attribute of the assertion
#
#         """
#         assert self.margin > 0, f"Margin {self.margin} is nonpositive"
#         if data is not None:
#             sample_size = self.test.sample_size(
#                 data,
#                 alpha=self.contest.risk_limit,
#                 reps=reps,
#                 prefix=prefix,
#                 quantile=quantile,
#                 seed=seed,
#             )
#         else:
#             """
#             Construct data.
#             For POLLING, values are 0, 1/2, and u.
#             For CARD_COMPARISON, values are overstatement assorter values corresponding to
#             overstatements of 2u (at rate_2), u (at rate_1), or 0.
#             """
#             big = (
#                 self.assorter.upper_bound
#                 if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING
#                 else self.make_overstatement(overs=0)
#             )
#             small = (
#                 0
#                 if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING
#                 else self.make_overstatement(overs=1/2)
#             )
#             rate_1 = (
#                 rate_1 if rate_1 is not None else (1 - self.margin) / 2
#             )  # rate of small values
#             x = big * np.ones(self.test.N)
#             if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING:
#                 if self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.IRV:
#                     raise NotImplementedError(
#                         f"data must be provided to estimate sample sizes for IRV assertions"
#                     )
#                 else:  # get tally
#                     if self.contest.tally:
#                         n_0 = self.contest.tally[self.loser]
#                         n_big = self.contest.tally[self.winner]
#                         n_half = self.test.N - n_0 - n_big
#                         x = interleave_values(n_0, n_half, n_big, big=big)
#                     else:
#                         raise ValueError(
#                             f"contest {self.contest} tally required but not defined"
#                         )
#             elif (
#                 self.contest.audit_type in [Audit.AUDIT_TYPE.CARD_COMPARISON, Audit.AUDIT_TYPE.ONEAUDIT]
#             ):  # comparison audit
#                 rate_1_i = (
#                     np.arange(0, self.test.N, step=int(1 / rate_1), dtype=int)
#                     if rate_1
#                     else []
#                 )
#                 rate_2_i = (
#                     np.arange(0, self.test.N, step=int(1 / rate_2), dtype=int)
#                     if rate_2
#                     else []
#                 )
#                 x[rate_1_i] = small
#                 x[rate_2_i] = 0
#             else:
#                 raise NotImplementedError(
#                     f"audit type {self.contest.audit_type} for contest {self.contest} not implemented"
#                 )
#             sample_size = self.test.sample_size(
#                 x,
#                 alpha=self.contest.risk_limit,
#                 reps=reps,
#                 prefix=prefix,
#                 quantile=quantile,
#                 seed=seed,
#             )
#         self.sample_size = sample_size
#         return sample_size
#
#     @classmethod
#     def interleave_values(
#         cls,
#         n_small: int,
#         n_med: int,
#         n_big: int,
#         small: float = 0,
#         med: float = 1/2,
#         big: float = 1,
#     ):
#         r"""
#         make an interleaved population of n_s values equal to small, n_m values equal to med, and n_big equal to big
#         Start with a small if n_small > 0
#         """
#         N = n_small + n_med + n_big
#         x = np.zeros(N)
#         i_small = 0
#         i_med = 0
#         i_big = 0
#         r_small = 1 if n_small else 0
#         r_med = 1 if n_med else 0
#         r_big = 1
#         if r_small:  # start with small
#             x[0] = small
#             i_small = 1
#             r_small = (n_small - i_small) / n_small
#         elif r_med:  # start with 1/2
#             x[0] = med
#             i_med = 1
#             r_med = (n_med - i_med) / n_med
#         else:
#             x[0] = big
#             i_big = 1
#             r_big = (n_big - i_big) / n_big
#         for i in range(1, N):
#             if r_small > r_big:
#                 if r_med > r_small:
#                     x[i] = med
#                     i_med += 1
#                     r_med = (n_med - i_med) / n_med
#                 else:
#                     x[i] = small
#                     i_small += 1
#                     r_small = (n_small - i_small) / n_small
#             elif r_med > r_big:
#                 x[i] = med
#                 i_med += 1
#                 r_med = (n_med - i_med) / n_med
#             else:
#                 x[i] = big
#                 i_big += 1
#                 r_big = (n_big - i_big) / n_big
#         return x
#
#     @classmethod
#     def make_supermajority_assertion(
#         cls,
#         contest: object=None,
#         share_to_win: float = 1/2,
#         winner: str = None,
#         loser: list = None,
#         test: callable = None,
#         test_kwargs: dict = {},
#         estim: callable = None,
#         bet: callable = None,
#     ):
#         """
#         Construct assertion that winner got >= share_to_win \in (0,1) of the valid votes
#
#         **TODO: This method assumes there was a winner. To audit that there was no winner requires
#         flipping things.**
#
#         An equivalent condition is:
#
#         (votes for winner)/(2*share_to_win) + (invalid votes)/2 > 1/2.
#
#         Thus the correctness of a super-majority outcome--where share_to_win >= 1/2--can
#         be checked with a single assertion.
#
#         share_to_win < 1/2 might be useful for some social choice functions, including
#         primaries where candidates who receive less than some threshold share are
#         eliminated.
#
#         A CVR with a mark for more than one candidate in the contest is considered an
#         invalid vote.
#
#         Parameters
#         -----------
#         contest:
#             contest object instance to which the assertion applies
#         share_to_win: float
#             fraction of the valid votes the winner must get to win
#         winner:
#             identifier of winning candidate
#         loser: list
#             list of identifiers of losing candidate(s)
#         test: instance of NonnegMean
#             risk function for the contest
#         test_kwargs: dict
#             kwargs for the test
#         estim: an estimation method of NonnegMean
#             estimator the alpha_mart test uses for the alternative
#         bet: method to choose the bet for betting_mart risk function
#
#         Returns
#         --------
#         a dict containing one Assertion
#
#         """
#         assertions = {}
#         wl_pair = winner + " v " + Contest.CANDIDATES.ALL_OTHERS
#         cands = loser.copy()
#         cands.append(winner)
#         _test = NonnegMean(
#             test=test,
#             estim=estim,
#             bet=bet,
#             u=1 / (2 * contest.share_to_win),
#             N=contest.cards,
#             t=1 / 2,
#             random_order=True,
#             **test_kwargs
#         )
#         assertions[wl_pair] = Assertion(
#             contest,
#             winner=winner,
#             loser=Contest.CANDIDATES.ALL_OTHERS,
#             assorter=Assorter(
#                 contest=contest,
#                 assort=lambda c, contest_id=contest.id: (
#                     CVR.as_vote(c.get_vote_for(contest.id, winner))
#                     / (2 * contest.share_to_win)
#                     if c.has_one_vote(contest.id, cands)
#                     else 1 / 2
#                 ),
#                 upper_bound=1 / (2 * contest.share_to_win),
#             ),
#             test=_test,
#             estim=estim,
#             bet=bet,
#             test_kwargs = test_kwargs
#         )
#         return assertions
#
#     @classmethod
#     def set_all_margins_from_cvrs(
#         cls,
#         audit: object = None,
#         contests: dict = None,
#         cvr_list: "Collection[CVR]" = None,
#     ):
#         """
#         Find all the assorter margins in a set of Assertions. Updates the dict of dicts of assertions
#         and the contest dict.
#
#         Appropriate only if cvrs are available. Otherwise, base margins on the reported results.
#
#         This function is primarily about side-effects on the assertions in the contest dict.
#
#         Parameters
#         ----------
#         audit: Audit
#             information about the audit
#         contests: dict of Contest objects
#         cvr_list: Collection
#             collection of CVR objects
#
#         Returns
#         -------
#         min_margin: float
#             smallest margin in the audit
#
#         Side effects
#         ------------
#         sets the margin of every assertion
#         sets the assertion.test.u for every assertion, according to whether
#            `assertion.contest.audit_type==Audit.AUDIT_TYPE.POLLING`
#            or `assertion.contest.audit_type in [Audit.AUDIT_TYPE.CARD_COMPARISON, Audit.AUDIT_TYPE.ONEAUDIT]`
#         """
#         min_margin = np.inf
#         for c, con in contests.items():
#             con.margins = {}
#             for a, asn in con.assertions.items():
#                 asn.set_margin_from_cvrs(audit, cvr_list)
#                 margin = asn.margin
#                 con.margins.update({a: margin})
#                 if con.audit_type == Audit.AUDIT_TYPE.POLLING:
#                     u = asn.assorter.upper_bound
#                 elif con.audit_type in [
#                     Audit.AUDIT_TYPE.CARD_COMPARISON,
#                     Audit.AUDIT_TYPE.ONEAUDIT,
#                 ]:
#                     u = 2 / (2 - margin / asn.assorter.upper_bound)
#                 else:
#                     raise NotImplementedError(
#                         f"audit type {con.audit_type} not implemented"
#                     )
#                 asn.test.u = u
#                 min_margin = min(min_margin, margin)
#         return min_margin
#
#     @classmethod
#     def set_p_values(
#         cls, contests: dict, mvr_sample: list, cvr_sample: list = None, use_all=False
#     ) -> float:
#         """
#         Find the p-value for every assertion and update assertions & contests accordingly
#
#         update p_value, p_history, proved flag, the maximum p-value for each contest.
#
#         Primarily about side-effects.
#
#         Parameters
#         ----------
#         contests: dict of dicts
#             the contest data structure. outer keys are contest identifiers; inner keys are assertions
#
#         mvr_sample: list of CVR objects
#             the manually ascertained voter intent from sheets, including entries for phantoms
#
#         cvr_sample: list of CVR objects
#             the cvrs for the same sheets, for ballot-level comparison audits
#             not needed for polling audits
#
#         Returns
#         -------
#         p_max: float
#             largest p-value for any assertion in any contest
#
#         Side-effects
#         ------------
#         Sets u for every test for every assertion, according to whether the corresponding audit method
#         is Audit.AUDIT_TYPE.CARD_COMPARISON, Audit.AUDIT_TYPE.ONEAUDIT, or Audit.AUDIT_TYPE.POLLING.
#         Sets contest max_p to be the largest P-value of any assertion for that contest
#         Updates p_value, p_history, and proved for every assertion
#
#         """
#         if cvr_sample is not None:
#             assert len(mvr_sample) == len(cvr_sample), "unequal numbers of cvrs and mvrs"
#         p_max = 0
#         for c, con in contests.items():
#             con.p_values = {}
#             con.proved = {}
#             contest_max_p = 0
#             for a, asn in con.assertions.items():
#                 d, u = asn.mvrs_to_data(mvr_sample, cvr_sample, use_all=use_all)
#                 asn.test.u = u  # set upper bound for the test for each assorter
#                 asn.p_value, asn.p_history = asn.test.test(d)
#                 asn.proved = (asn.p_value <= con.risk_limit) or asn.proved
#                 con.p_values.update({a: asn.p_value})
#                 con.proved.update({a: asn.proved})
#                 contest_max_p = np.max([contest_max_p, asn.p_value])
#             contests[c].max_p = contest_max_p
#             p_max = np.max([p_max, contests[c].max_p])
#         return p_max
#
#     @classmethod
#     def reset_p_values(cls, contests: dict) -> bool:
#         '''
#         Resets the p-values, p_history, and proved for every assertion
#         '''
#         for c, con in contests.items():
#             con.p_values = {}
#             con.proved = {}
#             for a, asn in con.assertions.items():
#                 asn.p_value, asn.p_history = 1, []
#                 asn.proved = False
#                 con.p_values.update({a: asn.p_value})
#                 con.proved.update({a: asn.proved})
#             contests[c].max_p = 1
#         return True
#
#
# class PluralityAssertion(Assertion):
#     pass
#
#
# class InstantRunoffAssertion(Assertion):
#     pass
#
#
# @dataclass
# class AssertionHandler(ABC):
#     con: Contest
#     registry: NodeRegistry = None
#     root: Node = None
#     proved: bool = False
#     stratified: bool = False  # todo add stratum object here
#
#     @abstractmethod
#     def initialize_nodes(self):
#         pass
#
#     @abstractmethod
#     def process_ballots(self, mvrs: Collection[CVR], cvrs: Collection[CVR] = None):
#         pass
#
#
# class PluralityAssertionHandler(AssertionHandler):
#     def __init__(self, con: contest.Contest,
#         winner: list=None,
#         loser: list=None,
#         test: callable=None,
#         test_kwargs: dict={},
#         estim: callable=None,
#         bet: callable=None,
#     ):
#         """
#         Construct assertions that imply the winner(s) got more votes than the loser(s).
#
#         The assertions are that every winner beat every loser: there are
#         len(winner)*len(loser) pairwise assertions in all.
#
#         Parameters
#         -----------
#         contest: instance of Contest
#             contest to which the assertions are relevant
#         winner: list
#             list of identifiers of winning candidate(s)
#         loser: list
#             list of identifiers of losing candidate(s)
#         test: NonnegMean
#             statistical test to use
#         test_kwargs: dict
#             kwargs for the test
#         estim: callable
#             estimator for alpha test supermartingale
#         bet: callable
#             bet for betting test supermartingale
#
#         Returns
#         --------
#         a dict of Assertions
#
#         """
#         winners = con.winner
#         losers = list(set(con.candidates) - set(winners))
#         test = con.test
#         test_kwargs = con.test_kwargs
#         estim = con.estim
#         bet = con.bet
#
#         assertions = {}
#         test = test if test is not None else con.test
#         estim = estim if estim is not None else con.estim
#         bet = bet if bet is not None else con.bet
#         for winr in winner:
#             for losr in loser:
#                 wl_pair = winr + " v " + losr
#                 _test = NonnegMean(
#                     test=test,
#                     estim=estim,
#                     bet=bet,
#                     g=contest.g,
#                     u=1,
#                     N=contest.cards,
#                     t=1 / 2,
#                     random_order=True,
#                     **test_kwargs
#                 )
#                 assorter = Assorter(
#                         contest=contest,
#                         assort=lambda c, contest_id=contest.id, winr=winr, losr=losr: (
#                             CVR.as_vote(c.get_vote_for(contest.id, winr))
#                             - CVR.as_vote(c.get_vote_for(contest.id, losr))
#                             + 1
#                         )
#                         / 2,
#                         upper_bound=1,
#                     )
#                 assertions[wl_pair] = PluralityAssertion(
#                     contest,
#                     winner=winr,
#                     loser=losr,
#                     assorter=assorter,
#                     test=_test,
#                 )
#
#
# class SupermajorityAssertionHandler(AssertionHandler):
#     pass
#
#
# class InstantRunoffAssertionHandler(AssertionHandler):
#     # @dataclass
#     # class node_signature:
#     #     order: list
#     #     winner: str
#     #     loser: str
#
#     def __init__(self, con: Contest):
#         super().__init__(contest=con)
#         self.registry = NodeRegistry()
#         if con.assertion_json:  # preload assertions
#             pass
#         else:  # create assertions from scratch
#             self.initialize_nodes()
#
#     def initialize_nodes(self):
#         ncand = len(set(self.con.candidates))
#         for order in permutations(range(ncand)):  # representation: order = [..., 3rd place, runner-up, winner]
#             if order[-1] == self.con.winner:
#                 continue
#             node = self.registry.get_composite_node(order)
#             node = Node(self.audit, np.array(order), self.weigher)
#             new_reqs = node.request_all_requirements(no_DNDs=self.req_no_dnds)
#             for new_req in new_reqs:
#                 node.add_to_watchlist(self.reqs.get(new_req))
#             node.queue_counter = self.next_queue_count()
#             self.nodes.put(node)
#         self.stat_max_depth = ncand
#
#         self.root = Node(
#             contest=self.con,
#             node_type=Node.NODE_TYPE.ROOT,
#             node_id=self.con.id,
#             parent=None,
#             children=[],
#             assertion_handler=self,
#         )
#         self.registry.add_node(self.root)
#
#     @classmethod
#     def make_assertions_from_json(
#         cls,
#         contest: object = None,
#         candidates: list = None,
#         json_assertions: dict = None,
#         test: callable = None,
#         test_kwargs: dict = {},
#         estim: callable = None,
#         bet: callable = None,
#     ):
#         """
#         dict of Assertion objects from a RAIRE-style json representations of assertions.
#
#         The assertion_type for each assertion must be one of the JSON_ASSERTION_TYPES
#         (class constants).
#
#         Parameters
#         ----------
#         contest: Contest instance
#             contest to which the assorter applies
#         candidates:
#             list of identifiers for all candidates in relevant contest.
#         json_assertions:
#             Assertions to be tested for the relevant contest.
#         test: instance of NonnegMean
#             risk function for the contest
#         test_kwargs: dict
#             kwargs for the test
#         estim: an estimation method of NonnegMean
#             estimator the test uses for the alternative
#
#         Returns
#         -------
#         dict of assertions for each assertion specified in 'json_assertions'.
#         """
#         assertions = {}
#         for assrtn in json_assertions:
#             winr = assrtn["winner"]
#             losr = assrtn["loser"]
#             if assrtn["assertion_type"] == cls.WINNER_ONLY:
#                 # CVR is a vote for the winner only if it has the
#                 # winner as its first preference
#                 winner_func = lambda v, contest_id=contest.id, winr=winr: (
#                     1 if v.get_vote_for(contest_id, winr) == 1 else 0
#                 )
#
#                 # CVR is a vote for the loser if they appear and the
#                 # winner does not, or they appear before the winner
#                 loser_func = lambda v, contest_id=contest.id, winr=winr, losr=losr: v.rcv_lfunc_wo(
#                     contest_id, winr, losr
#                 )
#
#                 wl_pair = winr + " v " + losr
#                 _test = NonnegMean(
#                     test=test,
#                     estim=estim,
#                     bet=bet,
#                     u=1,
#                     N=contest.cards,
#                     t=1 / 2,
#                     random_order=True,
#                     **test_kwargs
#                 )
#                 assertions[wl_pair] = Assertion(
#                     contest,
#                     Assorter(
#                         contest=contest,
#                         winner=winner_func,
#                         loser=loser_func,
#                         upper_bound=1,
#                     ),
#                     winner=winr,
#                     loser=losr,
#                     test=_test,
#                 )
#
#             elif assrtn["assertion_type"] == cls.IRV_ELIMINATION:
#                 # Context is that all candidates in 'eliminated' have been
#                 # eliminated and their votes distributed to later preferences
#                 elim = [e for e in assrtn["already_eliminated"]]
#                 remn = [c for c in candidates if c not in elim]
#                 # Identifier for tracking which assertions have been proved
#                 wl_given = winr + " v " + losr + " elim " + " ".join(elim)
#                 _test = NonnegMean(
#                     test=test,
#                     estim=estim,
#                     bet=bet,
#                     u=1,
#                     N=contest.cards,
#                     t=1 / 2,
#                     random_order=True,
#                     **test_kwargs
#                 )
#                 assertions[wl_given] = Assertion(
#                     contest,
#                     Assorter(
#                         contest=contest,
#                         assort=lambda v, contest_id=contest.id, winner=winr, loser=losr, remn=remn: (
#                             v.rcv_votefor_cand(contest.id, winner, remn)
#                             - v.rcv_votefor_cand(contest.id, loser, remn)
#                             + 1
#                         )
#                         / 2,
#                         upper_bound=1,
#                     ),
#                     winner=winr,
#                     loser=losr,
#                     test=_test,
#                 )
#             else:
#                 raise NotImplemented(
#                     f'JSON assertion type {assrtn["assertion_type"]} not implemented.'
#                 )
#         return assertions
#
#
# def make_all_assertions(contests: dict):
#     """
#     Construct all the assertions to audit the contests and add the assertions to the contest dict
#
#     Parameters
#     ----------
#     contests: dict
#         dict of Contest objects
#
#     Returns
#     -------
#     True
#
#     Side Effects
#     ------------
#     creates assertions and adds the dict of assertions relevant to each contest to the contest
#     object's `assertions` attribute
#
#     """
#     for c, con in contests.items():
#         scf = con.choice_function
#         winrs = con.winner
#         losrs = list(set(con.candidates) - set(winrs))
#         # test = con.test
#         # test_kwargs = con.test_kwargs
#         # estim = con.estim
#         # bet = con.bet
#         if isinstance(scf, contest.Plurality):
#             contests[c].assertion_handler = PluralityAssertionHandler(con=con, winner=winrs, loser=losrs)
#         elif isinstance(scf, contest.Supermajority):
#             contests[c].assertion_handler = SupermajorityAssertionHandler(con=con)
#         elif isinstance(scf, contest.InstantRunoff):
#             # Assumption: contests[c].assertion_json yields list assertions in JSON format.
#             contests[c].assertion_handler = InstantRunoffAssertionHandler(con=con)
#         else:
#             raise NotImplementedError(
#                 f"Social choice function {scf} is not implemented."
#             )
#     return True
#
#
# class Assorter:
#     """
#     Class for generic Assorter.
#
#     An assorter must either have an `assort` method or both `winner` and `loser` must be defined
#     (in which case assort(c) = (winner(c) - loser(c) + 1)/2. )
#
#     Class parameters:
#     -----------------
#     contest: Contest instance
#         the contest to which this Assorter applies
#
#     winner: callable
#         maps a dict of selections into the value 1 if the dict represents a vote for the winner
#
#     loser : callable
#         maps a dict of selections into the value 1 if the dict represents a vote for the winner
#
#     assort: callable
#         maps dict of selections into float
#
#     upper_bound: float
#         a priori upper bound on the value the assorter assigns to any dict of selections
#
#     tally_pool_means: dict
#         mean of the assorter over each tally_pool of CVRs, for ONEAudit
#
#     The basic method is assort, but the constructor can be called with (winner, loser)
#     instead. In that case,
#
#         assort = (winner - loser + 1)/2
#
#     """
#
#     def __init__(
#         self,
#         contest: object = None,
#         assort: callable = None,
#         winner: str = None,
#         loser: str = None,
#         upper_bound: float = 1,
#         tally_pool_means: dict = None,
#     ):
#         """
#         Constructs an Assorter.
#
#         If assort is defined and callable, it becomes the class instance of assort
#
#         If assort is None but both winner and loser are defined and callable,
#            assort is defined to be 1/2 if winner=loser; winner, otherwise
#
#
#         Parameters
#         -----------
#         contest: Contest instance
#             the contest to which the assorter is relevant
#         assort: callable
#             maps a dict of votes into [0, upper_bound]
#         winner: callable
#             maps a pattern into [0, 1]
#         loser: callable
#             maps a pattern into [0, 1]
#         upper_bound: float > 0
#             a priori upper bound on the value the assorter can take
#         tally_pool_means: dict
#             dict of the mean value of the assorter over each tally_pool of CVRs
#
#         """
#         self.contest = contest
#         self.winner = winner
#         self.loser = loser
#         self.upper_bound = upper_bound
#         if assort is not None:
#             assert callable(assort), "assort must be callable"
#             self.assort = assort
#         else:
#             assert callable(winner), "winner must be callable if assort is None"
#             assert callable(loser), "loser must be callable if assort is None"
#             self.assort = lambda cvr: (self.winner(cvr) - self.loser(cvr) + 1) / 2
#         self.tally_pool_means = tally_pool_means
#
#     def __str__(self):
#         """
#         string representation
#         """
#         return (
#             f"contest_id: {self.contest.id}\nupper bound: {self.upper_bound}, "
#             + f"winner defined: {callable(self.winner)}, loser defined: {callable(self.loser)}, "
#             + f"assort defined: {callable(self.assort)} "
#             + f"tally_pool_means: {bool(self.tally_pool_means)}"
#         )
#
#     def mean(self, cvr_list: "Collection[CVR]" = None, use_style: bool = True):
#         """
#         find the mean of the assorter applied to a list of CVRs
#
#         Parameters
#         ----------
#         cvr_list: Collection
#             a collection of cast-vote records
#         use_style: Boolean
#             does the audit use card style information? If so, apply the assorter only to CVRs
#             that contain the contest in question.
#
#         Returns
#         -------
#         mean: float
#             the mean value of the assorter over the collection of cvrs. If use_style, ignores CVRs that
#             do not contain the contest.
#         """
#         if use_style:
#             filtr = lambda c: c.has_contest(self.contest.id)
#         else:
#             filtr = lambda c: True
#         return np.mean([self.assort(c) for c in cvr_list if filtr(c)])
#
#     def set_tally_pool_means(
#         self,
#         cvr_list: "Collection[CVR]" = None,
#         tally_pools: Collection = None,
#         use_style: bool = True,
#     ):
#         """
#         create dict of pool means for the assorter from a set of CVRs
#
#         Parameters
#         ----------
#         cvr_list: Collection
#             cvrs from which the sample will be drawn
#
#         tally_pools: Collection [optional]
#             the labels of the tally groups
#
#         Returns
#         -------
#         nothing
#
#         Side effects
#         ------------
#         sets self.tally_pool_means
#         """
#         if not tally_pools:
#             tally_pools = set(c.tally_pool for c in cvr_list if c.pool)
#         tally_pool_dict = {}
#         for p in tally_pools:
#             tally_pool_dict[p] = {}
#             tally_pool_dict[p]["n"] = 0
#             tally_pool_dict[p]["tot"] = 0
#         if use_style:
#             filtr = lambda c: c.has_contest(self.contest.id)
#         else:
#             filtr = lambda c: True
#         for c in [cvr for cvr in cvr_list if (filtr(cvr) and cvr.pool)]:
#             tally_pool_dict[c.tally_pool]["n"] += 1
#             tally_pool_dict[c.tally_pool]["tot"] += self.assort(c)
#         self.tally_pool_means = {}
#         for p in tally_pools:
#             self.tally_pool_means[p] = (
#                 np.nan
#                 if tally_pool_dict[p]["n"] == 0
#                 else tally_pool_dict[p]["tot"] / tally_pool_dict[p]["n"]
#             )
#
#     def sum(self, cvr_list: "Collection[CVR]" = None, use_style: bool = True):
#         """
#         find the sum of the assorter applied to a list of CVRs
#
#         Parameters
#         ----------
#         cvr_list: Collection of CVRs
#             a collection of cast-vote records
#         use_style: Boolean
#             does the audit use card style information? If so, apply the assorter only to CVRs
#             that contain the contest in question.
#
#         Returns
#         -------
#         sum: float
#             sum of the value of the assorter over a list of CVRs. If use_style, ignores CVRs that
#             do not contain the contest.
#         """
#         if use_style:
#             filtr = lambda c: c.has_contest(self.contest.id)
#         else:
#             filtr = lambda c: True
#         return np.sum([self.assort(c) for c in cvr_list if filtr(c)])
#
#     def overstatement(self, mvr, cvr, use_style=True):
#         """
#         overstatement error for a CVR compared to the human reading of the ballot
#
#         If use_style, then if the CVR contains the contest but the MVR does
#         not, treat the MVR as having a vote for the loser (assort()=0)
#
#         If not use_style, then if the CVR contains the contest but the MVR does not,
#         the MVR is considered to be a non-vote in the contest (assort()=1/2).
#
#         Phantom CVRs and MVRs are treated specially:
#             A phantom CVR is considered a non-vote in every contest (assort()=1/2).
#             A phantom MVR is considered a vote for the loser (i.e., assort()=0) in every
#             contest.
#
#         Parameters
#         ----------
#         mvr: Cvr
#             the manual interpretation of voter intent
#         cvr: Cvr
#             the machine-reported cast vote record
#
#         Returns
#         -------
#         overstatement: float
#             the overstatement error
#         """
#         # sanity check
#         if use_style and not cvr.has_contest(self.contest.id):
#             raise ValueError(
#                 f"use_style==True but {cvr=} does not contain contest {self.contest.id}"
#             )
#         # assort the MVR
#         mvr_assort = (
#             0
#             if
#                 mvr.phantom or (use_style and not mvr.has_contest(self.contest.id))
#             else
#                 self.assort(mvr)
#         )
#         # assort the CVR
#         cvr_assort = (
#             self.tally_pool_means[cvr.tally_pool]
#             if
#                 cvr.pool and self.tally_pool_means is not None
#             else
#                 int(cvr.phantom) / 2 + (1 - int(cvr.phantom)) * self.assort(cvr)
#         )
#         return cvr_assort - mvr_assort
