import json
import numpy as np
import csv
import argparse
from copy import deepcopy
import os
from pathlib import Path
from cryptorandom.cryptorandom import SHA256, int_from_hash

from shangrla.core.Audit import Audit #, Assertion, Contest, CVR
from shangrla.core.NonnegMean import NonnegMean
from shangrla.core.assertion import *
import shangrla.core.contest as contest

from shangrla.raire.raire_utils import load_contests_from_raire_raw
from shangrla.raire.raire import compute_raire_assertions
from shangrla.raire.sample_estimator import cp_estimate

import itertools

def get_samplesize_and_overstatements(overstatements, pvalues, pvalue_threshold):
    below_thres = pvalues <= pvalue_threshold
    certified_thres = True
    where_thres = np.argmax(below_thres) + 1
    if sum(below_thres) == 0 or pvalues[where_thres - 1] == 0.0:
        certified_thres = False
        where_thres = len(pvalues)
    type_thres, counts_thres = np.unique(overstatements[:where_thres], return_counts=True)
    counts_thres = dict(zip([int(x) for x in type_thres], [int(x) for x in counts_thres]))
    counts_thres.setdefault(-2, 0)
    counts_thres.setdefault(-1, 0)
    counts_thres.setdefault(0, 0)
    counts_thres.setdefault(1, 0)
    counts_thres.setdefault(2, 0)
    sample_size = where_thres
    is_certified = certified_thres
    overstatement_counts = counts_thres
    return sample_size, is_certified, overstatement_counts


def merge_pvalues(assertions_dict):
    pvalue_histories = []
    for asrtn in assertions_dict:
        a = assertions_dict[asrtn]
        phist = a.p_history
        phist_running_min = np.minimum.accumulate(phist)
        pvalue_histories.append(phist_running_min)
    pvalue_histories_stacked = np.stack(pvalue_histories)
    pvalue_histories_merged  = np.amax(pvalue_histories_stacked, axis=0)
    return pvalue_histories_merged


def shuffle(cvrs, ordering):
    cvrs_shuffled = [cvrs[i-1] for i in ordering]
    return cvrs_shuffled


def read_election_files(datafile, marginfile, orderfile):
    rairedata_pre = "1\nContest,"
    rairedata = ""

    ballotnmbr = 0
    with open(datafile, "r") as file:
        line = file.readline()
        candmap = dict()
        candlist = [i.strip() for i in line.split(",")]
        for i, cand in enumerate(candlist):
            candmap[cand] = i
        ncand = len(candlist)

        # remove headers
        while True:
            if "-" in file.readline():
                break

        for line in file:
            f = line.split(" : ")
            strballot = f[0].split("(")[1].split(")")[0].split(",")
            if len(strballot) == 2 and strballot[1] == '':
                strballot = [strballot[0]]  # compatibility issue fix with trailing comma
            if len(strballot) == 1 and strballot[0] == '':
                strballot = []
            ballot = [candmap[i.strip()] for i in strballot]
            nvotes = int(f[1])
            for _ in range(nvotes):
                ballotnmbr += 1
                seq = ",".join([str(i) for i in ballot])
                rairedata += f"\n1,{ballotnmbr},{seq}"
    nballots = ballotnmbr

    margindata = [None] * ncand
    try:
        with open(marginfile, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                if csv_reader.line_num == 1: continue
                margindata[candmap[row[1].strip()]] = int(row[2]) / nballots
    except FileNotFoundError:
        pass
    margin = max(margindata, default=None)
    winner = int(np.argmax(np.array(margindata)))

    orderdata = []
    with open(orderfile, "r") as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if csv_reader.line_num == 1: continue
            orderdata.append([int(i) for i in row])

    rairedata_pre += f"1,{ncand}," + ",".join([str(i) for i in range(ncand)]) + f",winner,{winner}"
    rairedata = rairedata_pre + rairedata

    return ncand, winner, nballots, margin, orderdata, rairedata


# create audit object
audit = Audit.from_dict({
        'strata': {'stratum1': {'use_style': True,
                                'replacement': False}}
    })

# create contest object
datafile = "data/albury-ballots.txt"
margin = "data/albury-margins.csv"
orderings = "data/albury-orderings.csv"

ncand, winner, pop_size, margin, orderdata, data = read_election_files(datafile, margin, orderings)

name = "NSW2015_Albury"

raire_contests, cvrs = load_contests_from_raire_raw(data)
raire_contest = raire_contests[0]
cvr_input = [{"id": c, "votes": {'1': {i: cvrs[c]['1'][i] + 1 for i in cvrs[c]['1'].keys()}}} for c in cvrs.keys()]
assertions = compute_raire_assertions(raire_contest, cvrs, str(winner), cp_estimate, log=False)
assertions = [a.to_json() for a in assertions]

contest_dict = {'1': {'name': '1',
                      'risk_limit': 0.05,
                      'cards': pop_size,
                      'choice_function': contest.InstantRunoff(),
                      # 'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.IRV,
                      'n_winners': 1,
                      'candidates': [str(i) for i in range(ncand)],
                      'winner': [str(winner)],
                      'assertion_file': "./assertions_temp.json",
                      # 'assertion_json': assertions,
                      'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
                      'test': NonnegMean.alpha_mart,
                      'estim': NonnegMean.optimal_comparison
                      }
                }
cons = Contest.from_dict_of_dicts(contest_dict)

# Construct the dict of dicts of assertions for each contest.
# Assertion.make_all_assertions(cons)
make_all_assertions(cons)
audit.check_audit_parameters(cons)

# Calculate margins for each assertion.
cvr_list = CVR.from_dict(cvr_input)
min_margin = cons['1'].assertions.set_all_margins_from_cvrs(audit, cvr_list)
# min_margin = Assertion.set_all_margins_from_cvrs(audit, cons, cvr_list)
# hardest_assertion_name, hardest_assertion = min(cons['1'].assertions.items(), key=lambda x: x[1].margin)
#
# # Hack, set attributes for tests and estims
# for contest in cons.values():
#     for assertion in contest.assertions.values():
#         assertion.test.error_rate_2 = 1e-5
#
# # Calculate all the p-values.
cons['1'].assertions.set_p_values(cvr_list, cvr_list, use_all=True)
# Assertion.set_p_values(cons, cvr_list, cvr_list, use_all=True)
#pvalues = merge_pvalues(cons['1'].assertions.assertions)
cons['1'].assertions.registry.root[200]
pvalues = cons['1'].assertions.registry.root.eprocess.p_history()
# overstatements = np.array([cons['1'].assertions.registry.root.overstatement(cvr_list[i], cvr_list[i]) * 2
#                            for i in range(len(cvr_list))])

samplesize_5pct, certified_5pct, counts_5pc = get_samplesize_and_overstatements([],
                                                                                pvalues,
                                                                                0.05)

# print
print(f"{name = }\n"
      f"{pop_size = }\n"
      f"{margin = }\n"
      f"{min_margin = }\n"
      f"{samplesize_5pct = }\n"
      f"{certified_5pct = }\n"
      # f"{counts_5pc[2]}, {counts_5pc[1]}, {counts_5pc[-1]}, {counts_5pc[-2]}"
      )


