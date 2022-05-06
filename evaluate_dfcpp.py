import os, json
import shutil
import re
import numpy as np
import argparse
import pandas as pd
from sbfl.base import SBFL
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from sbfl.utils import read_dfcpp_coverage, read_dfcpp_test_results
from sbfl.utils import get_sbfl_scores_from_frame
from sbfl.score_aggregation import *


REMOVE_ORIGINAL_DATA = False

D4CPP_ROOT = os.path.abspath("../defects4cpp")
COVERAGE_DIR = os.path.join(D4CPP_ROOT, "coverage")
TAXONOMY_DIR = os.path.join(D4CPP_ROOT, "defects4cpp/taxonomy")
FORMULAE = ["Ochiai", "Jaccard", "Tarantula", "Op2", "GP13"]

# SBFL basic settings
TIE_BREAKERS = ["min", "average", "max"]
AGGR_SCHEMES = {
    "max_aggregation": max_aggregation,
    "mean_aggregation": mean_aggregation,
    "min_rank_based_voting": min_rank_based_voting,
    "dense_rank_based_voting": dense_rank_based_voting,
    "dense_rank_based_suspiciousness_aware_voting": dense_rank_based_suspiciousness_aware_voting,
    "dense_rank_based_tie_aware_voting": dense_rank_based_tie_aware_voting,
}


class Bug:
    def __init__(self, pid, vid):
        self.pid = pid
        self.vid = vid
        self.output_loaded = False

        # load metadata
        meta_path = os.path.join(TAXONOMY_DIR, self.pid, "meta.json")
        with open(meta_path, "r") as json_file:
            self.metadata = json.load(json_file)["defects"][int(vid) - 1]

    def __str__(self):
        return f"{self.pid}-{self.vid}"

    @property
    def id(self):
        return str(self)

    @property
    def num_test_cases(self):
        return self.metadata["num_cases"]

    @property
    def failing_tests_GT(self):
        return set([str(t) for t in self.metadata["case"]])

    @property
    def path_to_patch(self):
        str_vid = str(self.vid)
        file_name = f"{str_vid.zfill(4)}-buggy.patch"
        return os.path.join(TAXONOMY_DIR, self.pid, "patch", file_name)

    @property
    def buggy_files(self):
        files = set()
        with open(self.path_to_patch, "r") as f:
            for l in f:
                if l.startswith("+++"):
                    files.add(l.strip()[6:])
        return files

    @property
    def buggy_lines(self):
        lines = set()
        with open(self.path_to_patch, "r") as f:
            buggy_file, start_line = None, None
            for l in f:
                if l.startswith("+++"):
                    buggy_file = l.strip()[6:]
                    continue
                m = re.match("^@@ -\d+,\d+ \+(\d+),\d+ @@", l)
                if m:
                    start_line = int(m.group(1))
                    current_line = start_line
                    continue
                if l.rstrip() == "--":
                    buggy_file, start_line = None, None
                    continue
                if buggy_file and "test" not in buggy_file and start_line:
                    if l.startswith("-"):
                        # surrounding lines
                        lines.add((buggy_file, current_line - 1))
                        lines.add((buggy_file, current_line))
                    else:
                        if l.startswith("+"):
                            lines.add((buggy_file, current_line))
                        current_line += 1
        return lines

    @property
    def path_to_cov_df(self):
        return os.path.join(self.output_dir, "coverage_df.pkl")

    @property
    def path_to_test_results(self):
        return os.path.join(self.output_dir, "test_results.json")

    @property
    def passing_tests(self):
        return set(
            [t for t in self.test_results if self.test_results[t] == "passed"]
        )

    @property
    def failing_tests(self):
        return set(
            [t for t in self.test_results if self.test_results[t] == "failed"]
        )

    @property
    def source_files(self):
        return set(self.coverage_df.index.get_level_values("file"))

    @property
    def functions(self):
        return set(self.coverage_df.index.droplevel("line"))

    def load_output(self, output_dir):
        self.output_dir = output_dir
        self.coverage_df = None
        # load coverage
        if os.path.exists(self.path_to_cov_df) and os.path.exists(
            self.path_to_test_results
        ):
            self.coverage_df = pd.read_pickle(self.path_to_cov_df)
            with open(self.path_to_test_results, "r") as f:
                self.test_results = json.load(f)
            if self.num_test_cases != len(self.test_results):
                raise Exception(
                    f"Expected # tests: {self.num_test_cases} != Actual # tests: {len(self.test_results)}"
                )
        else:
            self.test_results = read_dfcpp_test_results(self.output_dir)
            if self.num_test_cases != len(self.test_results):
                raise Exception(
                    f"Expected # tests: {self.num_test_cases} != Actual # tests: {len(self.test_results)}"
                )
            self.coverage_df = read_dfcpp_coverage(
                self.output_dir,
                only_covered=True,
                verbose=True,
                encoding="ISO-8859-1",
            )
            # load test results
            bug.save_cov_df()
            with open(self.path_to_test_results, "w") as f:
                json.dump(self.test_results, f)

        if self.failing_tests != self.failing_tests_GT:
            raise Exception(
                f"Expected failings: {self.failing_tests_GT} != actual failings: {self.failing_tests}"
            )

        if len(self.source_files) == 0:
            raise Exception(f"Zero covered files")

        self.output_loaded = True

    def save_cov_df(self):
        self.coverage_df.astype(pd.SparseDtype("int", 0)).to_pickle(
            self.path_to_cov_df
        )

    def remove_cov_df(self):
        if os.path.exists(self.path_to_cov_df):
            os.remove(self.path_to_cov_df)


def load_bugs():
    errors = {}
    bugs = []
    # load bug objects
    for dir_name in sorted(os.listdir(COVERAGE_DIR)):
        path_to_output = os.path.join(COVERAGE_DIR, dir_name)
        if not os.path.isdir(path_to_output):
            # skip if the destination is not directory
            continue

        pid, vid = "_".join(dir_name.split("_")[:-2]), dir_name.split("_")[-2]

        print(pid, vid)

        # initialize a bug object
        bug = Bug(pid, vid)

        try:
            bug.load_output(path_to_output)
        except Exception as e:
            print(e)
            errors[bug.id] = str(e)
            continue

        bugs.append(bug)
        for test_dir in os.listdir(path_to_output):
            if not os.path.isdir(os.path.join(path_to_output, test_dir)):
                continue
            try:
                _pid = test_dir.split("-")[0]
                _vid = test_dir.split("-")[1].split("#")[1]
            except:
                continue
            if _pid == pid and _vid == vid:
                if os.path.exists(bug.path_to_cov_df) and os.path.exists(
                    bug.path_to_test_results
                ):
                    if REMOVE_ORIGINAL_DATA:
                        shutil.rmtree(os.path.join(path_to_output, test_dir))
        print(f"{bug} is added")

    bugs = sorted(bugs, key=lambda b: str(b))
    return bugs, errors


def localize_function(bugs, errors, result_dir):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    result_rows = []

    print("# bugs", len(bugs))
    for bug in bugs:
        try:
            print("=========================================================")
            print(bug)

            cov_df = bug.coverage_df
            failing_tests = bug.failing_tests

            print(
                f"(# lines, # tests, # failing tests): \
                {cov_df.shape[0], cov_df.shape[1], len(failing_tests)}"
            )

            buggy_lines = bug.buggy_lines
            buggy_files = set([f for f, l in buggy_lines])
            print(f"Buggy lines: {buggy_lines}")
            print(f"Buggy files: {buggy_files}")

            buggy_file_path_map = {}
            for buggy_file in buggy_files:
                for file_path in bug.source_files:
                    if file_path.endswith(buggy_file):
                        buggy_file_path_map[file_path] = buggy_file
                        break

            if len(buggy_files) != len(buggy_file_path_map):
                for buggy_file in buggy_files:
                    if buggy_file in buggy_file_path_map.values():
                        continue
                    for file_path in bug.source_files:
                        if os.path.basename(file_path) == os.path.basename(
                            buggy_file
                        ):
                            buggy_file_path_map[file_path] = buggy_file
                            break

            if len(buggy_files) != len(buggy_file_path_map):
                raise Exception(
                    f"The coverage of buggy files {buggy_files - set(buggy_file_path_map.values())} is not measured."
                )

            failure_coverage = cov_df[failing_tests]
            is_in_failure_coverage = np.any(failure_coverage, axis=1)
            files_covered_by_failure = set(
                failure_coverage.index[is_in_failure_coverage].get_level_values(
                    "file"
                )
            )

            print(
                f"# files covered by failure: {len(files_covered_by_failure)}"
            )
            buggy_files_not_covered_by_failure = (
                set(buggy_file_path_map) - files_covered_by_failure
            )

            print(
                f"# buggy files not covered by failure: {len(buggy_files_not_covered_by_failure)}"
            )
            if buggy_files_not_covered_by_failure:
                raise Exception(
                    f"Buggy files {buggy_files_not_covered_by_failure} is not covered by failing test casess"
                )

            buggy_components = set()
            buggy_functions = set()
            for component in bug.coverage_df.index:
                _filepath, _function, _line = component
                if (
                    _filepath in buggy_file_path_map
                    and (buggy_file_path_map[_filepath], _line) in buggy_lines
                ):
                    buggy_components.add(component)
                    if _function:
                        buggy_functions.add((_filepath, _function))
                    else:
                        raise Exception(
                            f"The function information is not available in coverage data for buggy line: {component}"
                        )

            if bug.coverage_df.index.isin(buggy_components).sum() == 0:
                raise Exception(f"{buggy_lines} are not in the coverage matrix")
            print("********************** Valid *************************")
            print("Buggy functions:", buggy_functions)
            with open(f"{result_dir}/{bug}_buggy_functions", "w") as f:
                f.write(
                    "\n".join(
                        [f"{file},{func}" for file, func in buggy_functions]
                    )
                )

            for formula in FORMULAE:
                line_scores = get_sbfl_scores_from_frame(
                    cov_df, failing_tests, sbfl=SBFL(formula=formula)
                )
                line_scores.to_pickle(f"{result_dir}/{bug}_{formula}_line.pkl")

                for scheme in AGGR_SCHEMES:
                    func_scores = AGGR_SCHEMES[scheme](
                        line_scores, level=["file", "function"]
                    )
                    func_scores = func_scores.sort_values(
                        by="score", ascending=False
                    )
                    func_scores["is_buggy_function"] = func_scores.index.isin(
                        buggy_functions
                    )
                    for tb in TIE_BREAKERS:
                        func_scores[("rank", tb)] = (
                            -func_scores["score"]
                        ).rank(method=tb)
                    func_scores.to_pickle(
                        f"{result_dir}/{bug}_{formula}_function_{scheme}.pkl"
                    )

                    for buggy_function in buggy_functions:
                        row = [
                            bug.id,
                            len(bug.source_files),
                            len(bug.functions),
                            formula,
                            scheme,
                            buggy_function,
                        ]
                        row += [
                            func_scores.loc[buggy_function][("rank", tb)]
                            for tb in TIE_BREAKERS
                        ]
                        result_rows.append(row)
        except Exception as e:
            print(e)
            errors[bug.id] = str(e)
            continue
    columns = [
        "bug",
        "num_total_files",
        "num_total_functions",
        "formula",
        "aggregation_scheme",
        "buggy_function",
    ] + [f"rank_{tb}" for tb in TIE_BREAKERS]
    return result_rows, columns, errors


def localize_file(bugs, errors, result_dir):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    result_rows = []

    print("# bugs", len(bugs))
    for bug in bugs:
        try:
            print("=========================================================")
            print(bug)

            cov_df = bug.coverage_df
            failing_tests = bug.failing_tests

            print(
                f"(# lines, # tests, # failing tests): \
                {cov_df.shape[0], cov_df.shape[1], len(failing_tests)}"
            )

            buggy_lines = bug.buggy_lines
            buggy_files = set([f for f, l in buggy_lines])
            print(f"Buggy lines: {buggy_lines}")
            print(f"Buggy files: {buggy_files}")

            buggy_file_path_map = {}
            for buggy_file in buggy_files:
                for file_path in bug.source_files:
                    if file_path.endswith(buggy_file):
                        buggy_file_path_map[file_path] = buggy_file
                        break

            if len(buggy_files) != len(buggy_file_path_map):
                for buggy_file in buggy_files:
                    if buggy_file in buggy_file_path_map.values():
                        continue
                    for file_path in bug.source_files:
                        if os.path.basename(file_path) == os.path.basename(
                            buggy_file
                        ):
                            buggy_file_path_map[file_path] = buggy_file
                            break

            if len(buggy_files) != len(buggy_file_path_map):
                raise Exception(
                    f"The coverage of buggy files {buggy_files - set(buggy_file_path_map.values())} is not measured."
                )

            failure_coverage = cov_df[failing_tests]
            is_in_failure_coverage = np.any(failure_coverage, axis=1)
            files_covered_by_failure = set(
                failure_coverage.index[is_in_failure_coverage].get_level_values(
                    "file"
                )
            )

            print(
                f"# files covered by failure: {len(files_covered_by_failure)}"
            )
            buggy_files_not_covered_by_failure = (
                set(buggy_file_path_map) - files_covered_by_failure
            )

            print(
                f"# buggy files not covered by failure: {len(buggy_files_not_covered_by_failure)}"
            )
            if buggy_files_not_covered_by_failure:
                raise Exception(
                    f"Buggy files {buggy_files_not_covered_by_failure} is not covered by failing test casess"
                )

            buggy_files = list(buggy_file_path_map.keys())
            with open(f"{result_dir}/{bug}_buggy_files", "w") as f:
                f.write("\n".join(buggy_files))

            for formula in FORMULAE:
                line_scores = get_sbfl_scores_from_frame(
                    cov_df, failing_tests, sbfl=SBFL(formula=formula)
                )
                line_scores.to_pickle(f"{result_dir}/{bug}_{formula}_line.pkl")

                for scheme in AGGR_SCHEMES:
                    file_scores = AGGR_SCHEMES[scheme](
                        line_scores, level="file"
                    )
                    file_scores = file_scores.sort_values(
                        by="score", ascending=False
                    )
                    file_scores["is_buggy_file"] = file_scores.index.isin(
                        buggy_files
                    )
                    for tb in TIE_BREAKERS:
                        file_scores[("rank", tb)] = (
                            -file_scores["score"]
                        ).rank(method=tb)
                    file_scores.to_pickle(
                        f"{result_dir}/{bug}_{formula}_file_{scheme}.pkl"
                    )

                    for buggy_file in buggy_files:
                        row = [
                            bug.id,
                            len(bug.source_files),
                            formula,
                            scheme,
                            buggy_file,
                        ]
                        row += [
                            file_scores.loc[buggy_file][("rank", tb)]
                            for tb in TIE_BREAKERS
                        ]
                        result_rows.append(row)
        except Exception as e:
            print(e)
            errors[bug.id] = str(e)
            continue
    columns = [
        "bug",
        "num_total_files",
        "formula",
        "aggregation_scheme",
        "buggy_file",
    ] + [f"rank_{tb}" for tb in TIE_BREAKERS]
    return result_rows, columns, errors


def draw_plot(df, level):
    import seaborn as sns
    import matplotlib.pyplot as plt

    tb = "max"
    N = [1, 3, 5, 10, 15, 30, 50, 100]
    for n in N:
        df[f"acc_{n}"] = df[f"rank_{tb}"] <= n
    sdf = df.groupby(["bug", "formula", "aggregation_scheme"]).max()
    sdf = sdf.groupby(["formula", "aggregation_scheme"]).sum().reset_index()
    sdf = sdf.melt(
        id_vars=["formula", "aggregation_scheme"],
        value_vars=[f"acc_{n}" for n in N],
        var_name="measure",
    )

    for formula, tdf in sdf.groupby("formula"):
        plt.figure(figsize=(12, 5))
        plt.title(f"{formula} (# total bugs = {df.bug.unique().shape[0]})")
        hue_order = (
            sdf[(sdf.measure == f"acc_{N[0]}") & (sdf.formula == formula)]
            .sort_values(by="value")
            .aggregation_scheme
        )
        ax = sns.barplot(
            data=tdf,
            x="measure",
            y="value",
            hue="aggregation_scheme",
            hue_order=hue_order,
        )
        for container in ax.containers:
            ax.bar_label(container)
        plt.legend(loc=(1, 0))
        plt.savefig(f"figures/{formula}_{level}.pdf", bbox_inches="tight")
        plt.show()


def export_output(result_rows, columns, output_file):
    df = pd.DataFrame(data=result_rows, columns=columns)
    df.to_csv(output_file, index=False)
    print(output_file)
    return df


def export_error(errors, error_file):
    error_df = pd.DataFrame.from_dict(
        errors, orient="index", columns=["error_meassage"]
    )
    error_df.index.name = "bug"
    error_df = error_df.reset_index().sort_values(by="bug")
    error_df.to_csv(error_file, index=False)
    print(error_file)
    return error_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", "-l", type=str, default="function", help="function or file"
    )
    args = parser.parse_args()
    level = args.level
    assert level in ["file", "function"]

    localizers = {"file": localize_file, "function": localize_function}

    bugs, errors = load_bugs()
    result_dir = f"./results/{level}"
    result_rows, columns, errors = localizers[level](bugs, errors, result_dir)

    print("Intermediate results", result_dir)

    df = export_output(result_rows, columns, f"./output_{level}.csv")
    draw_plot(df, level)

    export_error(errors, f"./errors_{level}.csv")
