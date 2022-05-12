import os
import pandas as pd

os.chdir("/home/coinse/Workspace/SBFL")
from evaluate_dfcpp import Bug


def load_sbfl_result():
    return pd.read_csv("output_function.csv")


sbfl_df = load_sbfl_result()


def get_sbfl_rank(pid, vid, sbfl_df, formula="Op2", aggs="max_aggregation"):
    bid = f"{pid}-{vid}"
    return sbfl_df[
        (sbfl_df.bug == bid)
        & (sbfl_df.formula == formula)
        & (sbfl_df.aggregation_scheme == aggs)
    ].rank_max.values[0]


def load_bug_data(pid, vid):
    bug = Bug(pid, vid)
    path_to_output = (
        f"/home/coinse/Workspace/defects4cpp/coverage/{pid}_{vid}_buggy"
    )
    bug.load_output(path_to_output)
    return bug


bugs = sorted([elem.split("-") for elem in set(sbfl_df.bug.values.tolist())])
bugs = [(tup[0], int(tup[1])) for tup in bugs]


num_tests = []
for bug in bugs:
    bid = f"{bug[0]}-{bug[1]}"
    num_test = load_bug_data(bug[0], bug[1]).num_test_cases
    num_tests.append([bid, num_test])
num_test_df = pd.DataFrame(num_tests, columns=["bug", "num_test"])


def get_num_test(pid, vid):
    bid = f"{pid}-{vid}"
    return num_test_df[num_test_df.bug == bid].num_test.values[0]


def get_tcs_cov(covmat, idx):
    cov_of_idx = covmat.loc[[idx]]
    return set(cov_of_idx.columns[cov_of_idx.values[0]])


def get_idx_covby(covmat, tcs, typ):
    sub_covmat = covmat[list(tcs)]
    if typ == "not":
        bool_vec = ~sub_covmat.any(axis=1)
    else:
        bool_vec = sub_covmat.all(axis=1)
    return set(covmat.index[bool_vec])


def find_covmat_idx(pid, covmat, fault_idx, debug):
    if pid in ["libucl", "wget2", "xbps", "yara"]:
        fault_idx = (fault_idx[0].split("/")[-1], fault_idx[1])
    cov_cands = covmat.index.values.tolist()
    matching_cands = [
        idx_tup
        for idx_tup in cov_cands
        if idx_tup[0].endswith(fault_idx[0]) and idx_tup[2] == fault_idx[1]
    ]
    if len(matching_cands) == 1:
        return (
            True,
            matching_cands[0],
        )  # (matching_cands[0][0], matching_cands[0][1])
    elif len(matching_cands) > 2:
        if debug:
            print("matching_cand:")
            for cand in sorted(matching_cands):
                print(f"{cand}")
            print(f"More than one candidate found for {fault_idx}")
        return False, f"More than one candidate found for {fault_idx}"
    else:
        match_file_cands = [
            idx_tup
            for idx_tup in cov_cands
            if idx_tup[0].endswith(fault_idx[0])
        ]
        if len(match_file_cands) == 0:
            if debug:
                print(f"No file level candidate found for {fault_idx}")
            return False, f"No file level candidate found for {fault_idx}"
        try:
            closest_less_cand = max(
                [cand for cand in match_file_cands if cand[2] < fault_idx[1]],
                key=lambda cand: cand[2],
            )
        except ValueError as e:
            if debug:
                print(
                    f"No less than candidate found for {fault_idx}; Need to debug"
                )
            return (
                False,
                f"No less than candidate found for {fault_idx}; Need to debug",
            )
        try:
            closest_greater_cand = min(
                [cand for cand in match_file_cands if cand[2] > fault_idx[1]],
                key=lambda cand: cand[2],
            )
        except ValueError as e:
            if debug:
                print(
                    f"No greater than candidate found for {fault_idx}; Need to debug"
                )
            return (
                False,
                f"No greater than candidate found for {fault_idx}; Need to debug",
            )
        if closest_less_cand[1] == closest_greater_cand[1]:
            if debug:
                print(
                    f"{fault_idx} is in the middle of {closest_less_cand} and {closest_greater_cand}"
                )
            return False, (
                closest_less_cand[0],
                closest_less_cand[1],
                fault_idx[1],
            )
        else:
            if debug:
                print("No exact match found")
            return False, "No exact match found"


def find_covmat_func(pid, covmat, fault_idx, debug):
    succ, ret = find_covmat_idx(pid, covmat, fault_idx, debug)
    if isinstance(ret, tuple):
        return True, (ret[0], ret[1])
    else:
        return succ, ret
