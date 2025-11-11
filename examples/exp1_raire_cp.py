import copy
import csv

import shangrla.core.contest as contest
from shangrla.core.assertion import *
from shangrla.raire.raire import compute_raire_assertions
from shangrla.raire.raire_utils import load_contests_from_raire_raw
from shangrla.raire.sample_estimator import cp_estimate

import warnings
warnings.filterwarnings("ignore")


def read_election_files(ballotfile, marginfile, orderfile):
    rairedata_pre = "1\nContest,"
    rairedata = ""

    ballotnmbr = 0
    with open(ballotfile, "r") as file:
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
            if csv_reader.line_num == 14: break  # Only used for debugging; TODO: remove this for real experiments
            #if csv_reader.line_num == 502: break  # Hacky way to set the number of replicate experiments
            orderdata.append([int(i) for i in row])

    rairedata_pre += f"1,{ncand}," + ",".join([str(i) for i in range(ncand)]) + f",winner,{winner}"
    rairedata = rairedata_pre + rairedata

    return ncand, winner, nballots, margin, orderdata, rairedata

def shuffle(cvrs, ordering):
    cvrs_shuffled = [cvrs[i-1] for i in ordering]
    return cvrs_shuffled

def get_samplesize(pvalues, pvalue_threshold):
    below_thres = pvalues <= pvalue_threshold
    where_thres = np.argmax(below_thres)
    if sum(below_thres) == 0 or pvalues[where_thres] == 0.0:
        is_certified = False
        where_thres = len(pvalues) - 1
    else:
        is_certified = True
    # `where_thres` is the sample size for the given set of p-values
    return where_thres, is_certified


# Define which contests to audit.
datafiles = ["Castle_Hill", "Cessnock", "Albury"]
datafile = datafiles[1]

# Define some auditing parameters.
datapath = "data-external/dirtree-elections-analysis/"
ballotfile = datapath +                "NSW2015/" + "Data_NA_" + datafile + ".txt_ballots" + ".txt"
marginfile = datapath + "margins/"   + "NSW2015/" + "Data_NA_" + datafile + ".txt_ballots" + ".csv"
orderfile  = datapath + "orderings/" + "NSW2015/" + "Data_NA_" + datafile + ".txt_ballots" + ".csv"

# Load data files.
ncand, winner, pop_size, margin, orderdata, rairedata = \
    read_election_files(ballotfile, marginfile, orderfile)

# Run RAIRE.
raire_contests, cvrs = load_contests_from_raire_raw(rairedata)
assertions = compute_raire_assertions(raire_contests[0], cvrs, str(winner), cp_estimate, log=False)
assertions = [a.to_json() for a in assertions]

# Create audit object.
audit = Audit.from_dict({
        'strata': {'stratum1': {'use_style': True,
                                'replacement': False}}
    })

# Create contest object.
contest_dict = {'1': {'name': '1',
                      'risk_limit': 0.05,
                      'cards': pop_size,
                      'choice_function': contest.InstantRunoff(),
                      'n_winners': 1,
                      'candidates': [str(i) for i in range(ncand)],
                      'winner': [str(winner)],
                      'assertion_json': assertions,
                      'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
                      'test': NonnegMean.alpha_mart,
                      #'estim': NonnegMean.optimal_comparison
                      'estim': NonnegMean.shrink_trunc,
                      'test_kwargs': {"d": 200}
                      }
                }
cons = Contest.from_dict_of_dicts(contest_dict)

# Set things up for auditing.
make_all_assertions(cons)
audit.check_audit_parameters(cons)

# Convert CVRs to a list of CVR objects.
cvr_input = [{"id": c, "votes": {'1': {i: cvrs[c]['1'][i] + 1 for i in cvrs[c]['1'].keys()}}} for c in cvrs.keys()]
cvr_list = CVR.from_dict(cvr_input)

# Run a simulated audit for each permutation provided.
pvalue_list = []
sampsize_list = []
cert_list = []
for (i, orderdata_i) in enumerate(orderdata):
    # Create a new copy of the objects created earlier, to use for the current simulated audit.
    current_contest = copy.deepcopy(cons['1'])

    # Apply a permutation to the CVRs.
    cvr_list_shuffled = shuffle(cvr_list, orderdata_i)
    #identity_permutation = list(range(pop_size + 1))[1:]
    #cvr_list_shuffled = shuffle(cvr_list, identity_permutation)

    # Calculate margins for each assertion (needed for a comparison audit).
    min_margin = current_contest.assertions.set_all_margins_from_cvrs(audit, cvr_list_shuffled)

    # Calculate all p-values (only for leaf nodes) for the given sample of ballots.
    # NOTE: currently assumes all CVRs are error-free.
    current_contest.assertions.set_p_values(cvr_list_shuffled, cvr_list_shuffled, use_all=True)

    # Cascade p-value calculation up to the root node.
    current_contest.assertions.registry.root[pop_size - 1]

    # Determine final sample size of audit.
    pvalues = current_contest.assertions.registry.root.eprocess.p_history()
    samplesize_5pct, certified_5pct = get_samplesize(pvalues,0.05)

    # Print some output.
    result_i = (datafile, pop_size, margin, i+1, samplesize_5pct, certified_5pct)
    print(datafile, pop_size, margin, i+1, samplesize_5pct, certified_5pct, sep=",")

    pvalue_list.append(pvalues)
    sampsize_list.append(samplesize_5pct)
    cert_list.append(certified_5pct)

# Saves data as a CSV file.
def write_csv(datalist, outfile):
    with open(outfile, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(datalist)

# Combine main summary metrics together.
summary_list = np.c_[sampsize_list, cert_list]

# Write some info to a file.
outfile1 = "output/exp1-" + datafile + "-raire-cp-summary1.csv"
outfile2 = "output/exp1-" + datafile + "-raire-cp-pvalues1.csv"
write_csv(summary_list, outfile1)
write_csv(pvalue_list, outfile2)
#np.savetxt(outfile1, summary_list, delimiter=',', header='sampsize,certified')
