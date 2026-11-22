import math
import numpy as np
import json
import csv
import types
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Collection
from typing import Tuple
from cryptorandom.cryptorandom import int_from_hash
from cryptorandom.sample import random_permutation
from cryptorandom.sample import sample_by_index
from .NonnegMean import NonnegMean


##########################################################################################
class Stratum:
    """
    stratum attributes
    """

    def __init__(
        self,
        id: str = None,
        max_cards: int = None,
        use_style: bool = None,
        replacement: bool = False,
        audit_type: str = None,
        test: callable = None,
        estimator: callable = None,
        bet: callable = None,
        test_kwargs: dict = {},
    ):
        self.id = id
        self.max_cards = max_cards
        self.use_style = use_style
        self.replacement = replacement
        self.audit_type = audit_type
        self.test = test
        self.estimator = estimator
        self.bet = bet
        self.test_kwargs = test_kwargs

    @classmethod
    def from_dict(cls, d: dict = None):
        s = Stratum()
        s.__dict__.update(d)
        return s


##########################################################################################
class NpEncoder(json.JSONEncoder):
    """
    for json dumps of Audit, Assertion, Contest, Stratum, FunctionType
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Assertion):
            return obj.to_dict()
        if isinstance(obj, Audit):
            return obj.__dict__
        if isinstance(obj, Contest):
            return obj.__dict__
        if isinstance(obj, Stratum):
            return obj.__dict__
        if isinstance(obj, types.FunctionType):
            return obj.__name__
        if isinstance(obj, NonnegMean):
            return str(obj)
        return super(NpEncoder, self).default(obj)

    @classmethod
    def trim_ints(cls, x):
        """
        turn int64 into an int

        Parameters
        ----------
        x: int64

        Returns
        -------
        int(x): int
        """
        if isinstance(x, np.int64):
            return int(x)
        else:
            raise TypeError


##########################################################################################
class Audit:
    """
    Various constants that specify what kind of contests are audited and how to audit them.
    Methods to estimate the sample size to audit every contest.
    Methods for logging and checking parameters.
    """

    ATTRIBUTES = (
        "seed",
        "sim_seed",
        "cvr_file",
        "manifest_file",
        "sample_file",
        "mvr_file",
        "log_file",
        "quantile",
        "error_rate_1",
        "error_rate_2",
        "reps",
        "max_cards",
        "strata",
    )

    class AUDIT_TYPE:
        """
        types of audit
        """

        AUDIT_TYPES = (
            POLLING := "POLLING",
            CARD_COMPARISON := "CARD_COMPARISON",
            ONEAUDIT := "ONEAUDIT",
        )
        # TODO: BATCH_COMPARISON, STRATIFIED, HYBRID, ...

    def __init__(
        self,
        seed: object = None,
        sim_seed: int = 123456789,
        cvr_file: str = None,
        manifest_file: str = None,
        sample_file: str = None,
        mvr_file: str = None,
        log_file: str = None,
        quantile: float = 0.9,
        error_rate_1: float = 0.001,
        error_rate_2: float = 0,
        reps: int = None,
        max_cards: int = None,
        strata: dict[str, Stratum] = None,
    ):
        self.seed = seed
        self.sim_seed = sim_seed
        self.cvr_file = (cvr_file,)
        self.manifest_file = manifest_file
        self.sample_file = sample_file
        self.mvr_file = mvr_file
        self.log_file = log_file
        self.quantile = quantile
        self.error_rate_1 = error_rate_1
        self.error_rate_2 = error_rate_2
        self.reps = reps
        self.max_cards = max_cards
        self.strata = strata

    def __str__(self):
        return str(self.__dict__)

    def find_sample_size(
        self,
        contests: dict = None,
        cvrs: list = None,
        mvr_sample: list = None,
        cvr_sample: list = None,
    ) -> int:
        """
        Estimate sample size for each contest and overall to allow the audit to complete.
        Uses simulations. For speed, uses the numpy.random Mersenne Twister instead of cryptorandom, a higher-quality
        PRNG used to select the actual audit sample.

        Parameters
        ----------
        contests: dict of dicts
            the contest data structure. outer keys are contest identifiers; inner keys are assertions
        cvrs: list of CVR objects
            the full set of CVRs
        mvr_sample: list of CVR objects
            manually ascertained votes
        cvr_sample: list of CVR objects
            CVRs corresponding to the cards that were manually inspected

        Returns
        -------
        new_size: int
            new sample size

        Side effects
        ------------
        sets c.sample_size for each Contest in contests
        if use_style, sets cvr.p for each CVR
        """
        # Currently, only unstratified audits are supported
        if len(self.strata) > 1:
            raise NotImplementedError("Stratified audits are not currently implemented.")
        stratum = next(iter(self.strata.values()))  # the only stratum
        if stratum.use_style and cvrs is None:
            raise ValueError("stratum.use_style==True but cvrs were not provided.")
        # unless style information is being used, the sample size is the same for every contest.
        old = 0 if stratum.use_style else len(mvr_sample)
        old_sizes = {c: old for c in contests.keys()}
        for c, con in contests.items():
            if stratum.use_style:
                old_sizes[c] = np.sum(
                    np.array([cvr.sampled for cvr in cvrs if cvr.has_contest(c)])
                )
            new_size = 0
            for a, asn in con.assertions.items():
                if not asn.proved:
                    if mvr_sample is not None:  # use MVRs to estimate the next sample size. Set `prefix=True` to use data
                        data, u = asn.mvrs_to_data(mvr_sample, cvr_sample)
                        new_size = max(
                            new_size,
                            asn.find_sample_size(
                                data=data,
                                prefix=True,
                                reps=self.reps,
                                quantile=self.quantile,
                                seed=self.sim_seed,
                            ),
                        )
                    else:
                        data = None
                        if con.audit_type == Audit.AUDIT_TYPE.ONEAUDIT:
                            if cvrs is None:
                                raise ValueError("ONEAudit sample size estimate requires cvrs.")
                            data, u = asn.mvrs_to_data(cvrs, cvrs, use_all=True)
                            # the following treatment of overstatement errors is a little crude because it assigns 
                            # errors to ONEAudit CVRs as if they were "raw" CVRs rather than averages,
                            # but it should be conservative as a result.
                            if self.error_rate_1:
                                idx = np.arange(0, len(data), math.floor(1/self.error_rate_1))
                                data[idx] = asn.make_overstatement(overs=1/2)
                            if self.error_rate_2:
                                idx = np.arange(0, len(data), math.floor(1/self.error_rate_2))
                                data[idx] = asn.make_overstatement(overs=1)
                        new_size = max(
                            new_size,
                            asn.find_sample_size(
                                data=data,
                                rate_1=self.error_rate_1,
                                rate_2=self.error_rate_2,
                                reps=self.reps,
                                quantile=self.quantile,
                                seed=self.sim_seed
                            )
                        )
            con.sample_size = new_size
        if stratum.use_style:
            for cvr in cvrs:
                if cvr.sampled:
                    cvr.p = 1
                else:
                    cvr.p = 0
                    for c, con in contests.items():
                        if cvr.has_contest(c) and not cvr.sampled:
                            cvr.p = max(
                                con.sample_size / (con.cards - old_sizes[c]), cvr.p
                            )
            total_size = math.ceil(np.sum([c.p for c in cvrs if not c.phantom]))
        else:
            total_size = np.max(
                np.array([con.sample_size for con in contests.values()])
            )
        return total_size

    def check_audit_parameters(self, contests: dict = None):
        """
        Check whether the audit parameters are valid; complain if not.

        Parameters
        ----------
        contests: dict of Contests
            contest-specific information for the audit

        Returns
        -------

        Side effects
        ------------
        raises exceptions if Audit parameters or contest parameters fail tests
        """
        assert (
            self.error_rate_1 >= 0
        ), "expected rate of 1-vote errors must be nonnegative"
        assert (
            self.error_rate_2 >= 0
        ), "expected rate of 2-vote errors must be nonnegative"
        for c, con in contests.items():
            assert (
                con.risk_limit > 0
            ), f"risk limit {con.risk_limit} negative in contest {c}"
            assert (
                con.risk_limit <= 1 / 2
            ), f"risk limit {con.risk_limit} exceeds 1/2 in contest {c}"
            # assert (
            #     con.choice_function in contest.SocialChoiceFunction.__subclasses__()
            # ), f"unsupported choice function {con.choice_function} in contest {c}"  # TODO this assertion is not needed?
            assert con.n_winners <= len(
                con.candidates
            ), f"more winners than candidates in contest {c}"
            assert (
                len(con.winner) == con.n_winners
            ), f"number of reported winners does not equal n_winners in contest {c}"
            for w in con.winner:
                assert (
                    w in con.candidates
                ), f"reported winner {w} is not a candidate in contest {c}"
            # if con.choice_function in [Contest.SOCIAL_CHOICE_FUNCTION.IRV,]:  # TODO fixme
            #     assert (
            #         con.n_winners == 1
            #     ), f"{con.choice_function} can have only 1 winner in contest {c}"
            # if con.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.IRV:
            #     assert con.assertion_file, f"IRV contest {c} requires an assertion file"

    def write_audit_parameters(self, contests: dict = None):
        """
        Write audit parameters as a json structure

        Parameters:
        ---------
        audit: Audit
            general information about the audit

        contests: dict of dicts
            contest-specific information for the audit

        Returns:
        --------
        no return value
        """
        log_file = self.log_file
        out = {"Audit": self, "contests": contests}
        with open(log_file, "w") as f:
            f.write(json.dumps(out, cls=NpEncoder))

    def summarize_status(self, contests: dict = None):
        """
        Determine whether the audit of individual assertions, contests, and the whole election are finished.

        Prints a summary.

        Parameters:
        -----------
        audit: Audit
            general information about the audit
        contests: dict of Contest objects
            dict of contest information

        Returns:
        --------
        done: boolean
            is the audit finished?
        """
        done = True
        for c, con in contests.items():
            print(f"\np-values for assertions in contest {c}")
            cpmax = 0
            for i, a in con.assertions.items():
                p_value = a.p_value()
                cpmax = np.max([cpmax, p_value])
                print(f"\t{i}: {p_value}")
            if cpmax <= contests[c].risk_limit:
                print(
                    f"\ncontest {c} AUDIT COMPLETE at risk limit {con.risk_limit}. Measured risk {cpmax}"
                )
            else:
                done = False
                print(
                    f"\ncontest {c} audit INCOMPLETE at risk limit {con.risk_limit}. Measured risk {cpmax}"
                )
                print("assertions remaining to be proved:")
                for i, a in con.assertions.items():
                    p_value = a.p_value()
                    if p_value > con.risk_limit:
                        print(f"\t{i}\t{a}: current risk {p_value}")
        return done

    @classmethod
    def from_dict(cls, d: dict = None):
        """
        make an Audit object from a dict of attributes
        assumes that the 'strata' attribute is itself a dict
        """
        a = Audit()
        # unpack stratum dictionary and create Stratum objects
        strat_obj = {}
        for id, st in d["strata"].items():
            strat_obj[id] = Stratum.from_dict(st)
            strat_obj[id].id = id
        d["strata"] = strat_obj
        a.__dict__.update(d)
        return a

