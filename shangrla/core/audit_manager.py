import numpy as np

from shangrla.core.assertion import make_all_assertions


class AuditManager:
    def __init__(self):
        self.contests = dict()
        self.audit = None # todo add support for multiple audit strategies
        self.min_margin = np.inf # todo add support for per-contest min margin

    def add_contest(self, contest):
        if contest.id in self.contests:
            raise ValueError(f"Contest ID already exists: {contest.id}")
        self.contests[contest.id] = contest

    def add_audit(self, audit):
        if self.audit is not None:
            raise ValueError(f"Multiple audits not yet supported!")
        self.audit = audit

    def make_all_assertions(self):
        make_all_assertions(self.contests)

    def check_audit_parameters(self):
        self.audit.check_audit_parameters(self.contests)

    def set_all_margins_from_cvrs(self, cvr_list):
        for cid, contest in self.contests.items():
            min_margin = contest.assertions.set_all_margins_from_cvrs(self.audit, cvr_list)
            if min_margin < self.min_margin:
                self.min_margin = min_margin

    def set_p_values(self, mvr_sample, cvr_sample):
        for cid, contest in self.contests.items():
            contest.assertions.set_p_values(mvr_sample, cvr_sample, use_all=True)

    def get_p_histories(self, start=0, end=None):
        pvalues = dict()
        print(self.contests.items())
        for cid, contest in self.contests.items():
            contest.assertions.registry.root[end]  # FIXME this should find the shortest endpoint that cross the risk
                                                   # limit if end=None
            pvalues[cid] = contest.assertions.registry.root.eprocess.p_history()
        return pvalues


