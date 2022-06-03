"""This module is used for summarizing the obtained results.

Summary
=======
This module provides methods for summarizing the sample statistics and model testing, evaluation, and benchmarking
results.

Functions
=========
- :func:`summarize_stats`: Summarize all obtained results and writes the summaries to CSV files.
- :func:`summarize_sampling`: Summarize the sampling statistics.
- :func:`summarize_testing`: Summarize the model testing results.
- :func:`summarize_evaluation`: Summarize the model evaluation results.
- :func:`summarize_benchmarking`: Summarize the model benchmarking results.
"""

import os

import numpy as np

from utils import load_seeds


def summarize_stats():
    """Summarize all obtained results and writes the summaries to CSV files."""

    out_dir = "summaries"
    os.makedirs(out_dir)

    summarize_sampling(out_dir)
    summarize_testing(out_dir)
    summarize_evaluation(out_dir)


def summarize_sampling(out_dir: str):
    """Summarize the sampling statistics.

    :param out_dir: The directory to store the summary in.
    """

    dims = {'setcov': '500r', 'combauc': '100i_500b', 'capfac': '100c_100f', 'indset': '500n'}

    # Collect all sample statistic files.
    train_stats = {}
    valid_stats = {}
    test_stats = {}
    for problem in dims.keys():
        train_stats[problem] = f"data/samples/{problem}/{dims[problem]}/train_stats.csv"
        valid_stats[problem] = f"data/samples/{problem}/{dims[problem]}/valid_stats.csv"
        test_stats[problem] = f"data/samples/{problem}/{dims[problem]}/test_stats.csv"
    set_files = {'train': train_stats, 'valid': valid_stats, 'test': test_stats}

    with open(os.path.join(out_dir, "sample_stats.csv"), 'w') as file:
        for dataset in set_files.keys():
            # Print headers.
            print(dataset, "setcov", "combauc", "capfac", "indset", sep=',', file=file)

            # Retrieve stats and print.
            files = set_files[dataset]
            _, setcov_total, setcov_unique = np.genfromtxt(files['setcov'], dtype=int, delimiter=',', skip_header=1)
            _, combauc_total, combauc_unique = np.genfromtxt(files['combauc'], dtype=int, delimiter=',', skip_header=1)
            _, capfac_total, capfac_unique = np.genfromtxt(files['capfac'], dtype=int, delimiter=',', skip_header=1)
            _, indset_total, indset_unique = np.genfromtxt(files['indset'], dtype=int, delimiter=',', skip_header=1)
            print("total", setcov_total, combauc_total, capfac_total, indset_total, sep=',', file=file)
            print("unique", setcov_unique, combauc_unique, capfac_unique, indset_unique, sep=',', file=file)
            print("", file=file)


def summarize_testing(out_dir: str):
    """Summarize the model testing results.

    :param out_dir: The directory to store the summary in.
    """

    train_seeds = load_seeds(name='train_seeds')[:2]

    # Collect all testing files.
    setcov_test = {}
    combauc_test = {}
    capfac_test = {}
    indset_test = {}
    for seed in train_seeds:
        setcov_test[str(seed)] = f"results/test/setcov/{seed}.csv"
        combauc_test[str(seed)] = f"results/test/combauc/{seed}.csv"
        capfac_test[str(seed)] = f"results/test/capfac/{seed}.csv"
        indset_test[str(seed)] = f"results/test/indset/{seed}.csv"
    problem_files = {'setcov': setcov_test, 'combauc': combauc_test, 'capfac': capfac_test, 'indset': indset_test}

    with open(os.path.join(out_dir, "test_stats.csv"), 'w') as file:
        for problem in problem_files.keys():
            # Print headers.
            print(problem, "25%", "50%", "75%", "100%", sep=',', file=file)

            # Retrieve stats and print.
            files = problem_files[problem]
            gcnn_fracs = np.zeros((len(train_seeds), 4))
            baseline_fracs = np.zeros((len(train_seeds), 4))
            for i in range(len(train_seeds)):
                stats = np.genfromtxt(files[str(train_seeds[i])], delimiter=',', skip_header=1)
                gcnn_fracs[i, :] = 100 * stats[0, 2:]
                baseline_fracs[i, :] = 100 * stats[1, 2:]
            gcnn_means = np.mean(gcnn_fracs, axis=0)
            gcnn_sds = np.std(gcnn_fracs, axis=0)
            baseline_means = baseline_fracs[0, :]  # Hybrid cut selector is deterministic.
            print("baseline", f"{baseline_means[0]:.2f}", f"{baseline_means[1]:.2f}", f"{baseline_means[2]:.2f}",
                  f"{baseline_means[3]:.2f}", sep=',', file=file)
            print("gcnn", f"{gcnn_means[0]:.2f} $\\pm$ {gcnn_sds[0]:.2f}",
                  f"{gcnn_means[1]:.2f} $\\pm$ {gcnn_sds[1]:.2f}", f"{gcnn_means[2]:.2f} $\\pm$ {gcnn_sds[2]:.2f}",
                  f"{gcnn_means[3]:.2f} $\\pm$ {gcnn_sds[3]:.2f}", sep=',', file=file)
            print("", file=file)


def summarize_evaluation(out_dir: str):
    """Summarize the model evaluation results.

    :param out_dir: The directory to store the summary in.
    """

    # Load the evaluation file (needs to be split up as NumPy arrays cannot have multiple dtypes).
    # [problem, difficulty, selector, status].
    string_stats = np.genfromtxt("results/eval.csv", dtype=str, delimiter=',', skip_header=1, usecols=(0, 1, 3, 9))

    # [instance, seed, n_nodes, n_lps].
    int_stats = np.genfromtxt("results/eval.csv", dtype=int, delimiter=',', skip_header=1, usecols=(2, 4, 5, 6))

    # [solve_time, gap, wall_time, process_time].
    float_stats = np.genfromtxt("results/eval.csv", dtype=float, delimiter=',', skip_header=1, usecols=(7, 8, 10, 11))

    # Filter stats.
    excluded = np.logical_not(np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit'))
    excluded_rows = np.arange(2, len(string_stats) + 2) * excluded  # Get row numbers in eval.csv.
    excluded_rows = excluded_rows[excluded_rows != 0]
    excluded_int = int_stats[excluded]
    excluded_string = string_stats[excluded]
    int_stats = int_stats[np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit')]
    float_stats = float_stats[np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit')]
    string_stats = string_stats[np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit')]

    # Record excluded [problem, difficulty, instance, seed] combinations.
    excluded = np.array(
        [[excluded_string[i, 0], excluded_string[i, 1], str(excluded_int[i, 0]), str(excluded_int[i, 1])] for i in
         range(len(excluded_string))])

    # Remove excluded entries.
    entries = np.array([[string_stats[i, 0], string_stats[i, 1], str(int_stats[i, 0]), str(int_stats[i, 1])] for i in
                        range(len(string_stats))])
    included_entries = np.array(len(entries) * [True])
    for i in range(len(entries)):
        entry = entries[i, :]
        for excluded_entry in excluded:
            if np.all(entry == excluded_entry):
                included_entries[i] = False
                break
    int_stats = int_stats[included_entries]
    float_stats = float_stats[included_entries]
    string_stats = string_stats[included_entries]

    # Split stats by problem, difficulty, and cut selector.
    problems = ['setcov', 'combauc', 'capfac', 'indset']
    difficulties = ['easy', 'medium', 'hard']
    selectors = ['hybrid', 'gcnn']
    split = {}
    for problem in problems:
        problem_int = int_stats[string_stats[:, 0] == problem]
        problem_float = float_stats[string_stats[:, 0] == problem]
        problem_string = string_stats[string_stats[:, 0] == problem]
        by_difficulty = {}
        for difficulty in difficulties:
            difficulty_int = problem_int[problem_string[:, 1] == difficulty]
            difficulty_float = problem_float[problem_string[:, 1] == difficulty]
            difficulty_string = problem_string[problem_string[:, 1] == difficulty]
            by_selector = {}
            for selector in selectors:
                selector_int = difficulty_int[difficulty_string[:, 2] == selector]
                selector_float = difficulty_float[difficulty_string[:, 2] == selector]
                selector_string = difficulty_string[difficulty_string[:, 2] == selector]
                by_selector[selector] = {'int': selector_int, 'float': selector_float, 'string': selector_string}
            by_difficulty[difficulty] = by_selector
        split[problem] = by_difficulty

    # Compute and record the 1-shifted geometric mean of solving time, and record the wins and node counts.
    with open(os.path.join(out_dir, "eval_stats.csv"), 'w') as file:
        # Print headers.
        print("", "", "Easy", "", "", "", "Medium", "", "", "", "Hard", "", sep=',', file=file)
        for problem in problems:
            # Print headers.
            headers = 3 * ["Time", "Wins", "Nodes", ""]
            print(problem, *headers, sep=',', file=file)

            lines = [[], []]
            for difficulty in difficulties:
                time_means = np.zeros(len(selectors))
                time_diffs = np.zeros(len(selectors))
                solve_times = np.zeros((len(split[problem][difficulty]['hybrid']['string']), len(selectors)))
                node_means = np.zeros(len(selectors), dtype=int)
                node_diffs = np.zeros(len(selectors))
                for i in range(len(selectors)):
                    selector = selectors[i]
                    int_data = split[problem][difficulty][selector]['int']
                    float_data = split[problem][difficulty][selector]['float']

                    # Compute 1-shifted geometric mean of solving time.
                    solve_time = float_data[:, 0]
                    k = len(solve_time)
                    s = 1
                    time_means[i] = np.power(np.prod(np.maximum(solve_time + s, 1)), 1 / k) - s
                    time_diffs[i] = 100 * np.sqrt(np.mean(np.power(solve_time - time_means[i], 2))) / time_means[i]

                    # Record solve times sorted by instance and seed (in that order).
                    sorted_float = float_data[np.lexsort((int_data[:, 1], int_data[:, 0]))]
                    solve_times[:, i] = sorted_float[:, 0]

                    # Compute the mean number of nodes.
                    node_counts = int_data[:, 2]
                    node_means[i] = np.round(np.mean(node_counts)).astype(int)
                    node_diffs[i] = 100 * np.round(np.std(node_counts)) / node_means[i]

                # Compute the number of wins for each selector.
                baseline_wins = solve_times[:, 0] <= solve_times[:, 1]
                gcnn_wins = 1 - baseline_wins
                baseline_wins = np.sum(baseline_wins)
                gcnn_wins = np.sum(gcnn_wins)
                lines[0] += [f"{time_means[0]:.2f} $\\pm$ {time_diffs[0]:.1f}%", f"{baseline_wins}",
                             f"{node_means[0]:d} $\\pm$ {node_diffs[0]:.1f}%", ""]
                lines[1] += [f"{time_means[1]:.2f} $\\pm$ {time_diffs[1]:.1f}%", f"{gcnn_wins}",
                             f"{node_means[1]:d} $\\pm$ {node_diffs[1]:.1f}%", ""]
            print("hybrid", *lines[0], sep=',', file=file)
            print("gcnn", *lines[1], sep=',', file=file)
            print("", file=file)

        # Print excluded results (if any).
        if len(excluded_rows) > 0:
            print("", file=file)
            print("Excluded", file=file)
            print("Row", "Problem", "Difficulty", "Instance", "Seed", sep=',', file=file)
            for i in range(len(excluded_rows)):
                print(excluded_rows[i], *excluded[i], sep=',', file=file)


def summarize_benchmarking(out_dir: str):
    """Summarize the benchmarking results.

    :param out_dir: The directory to store the summary in.
    """

    # Load the benchmarking file (needs to be split up as NumPy arrays cannot have multiple dtypes).
    # [problem, selector, instance, status].
    string_stats = np.genfromtxt("results/benchmark.csv", dtype=str, delimiter=',', skip_header=1, usecols=(0, 1, 3, 8))

    # [seed, n_nodes, n_lps].
    int_stats = np.genfromtxt("results/benchmark.csv", dtype=int, delimiter=',', skip_header=1, usecols=(2, 4, 5))

    # [solve_time, gap, wall_time, process_time].
    float_stats = np.genfromtxt("results/benchmark.csv", dtype=float, delimiter=',', skip_header=1,
                                usecols=(6, 7, 9, 10))

    # Filter stats.
    excluded = np.logical_not(np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit'))
    excluded_rows = np.arange(2, len(string_stats) + 2) * excluded  # Get row numbers in benchmark.csv.
    excluded_rows = excluded_rows[excluded_rows != 0]
    excluded_int = int_stats[excluded]
    excluded_string = string_stats[excluded]
    int_stats = int_stats[np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit')]
    float_stats = float_stats[np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit')]
    string_stats = string_stats[np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit')]

    # Record excluded [problem, instance, seed] combinations.
    excluded = np.array(
        [[excluded_string[i, 0], excluded_string[i, 2], str(excluded_int[i, 0])] for i in range(len(excluded_string))])

    # Remove excluded entries.
    entries = np.array(
        [[string_stats[i, 0], string_stats[i, 2], str(int_stats[i, 0])] for i in range(len(string_stats))])
    included_entries = np.array(len(entries) * [True])
    for i in range(len(entries)):
        entry = entries[i, :]
        for excluded_entry in excluded:
            if np.all(entry == excluded_entry):
                included_entries[i] = False
                break
    int_stats = int_stats[included_entries]
    float_stats = float_stats[included_entries]
    string_stats = string_stats[included_entries]

    # Split stats by problem and cut selector.
    problems = ['setcov', 'combauc', 'capfac', 'indset']
    selectors = ['hybrid', 'gcnn']
    split = {}
    for problem in problems:
        problem_int = int_stats[string_stats[:, 0] == problem]
        problem_float = float_stats[string_stats[:, 0] == problem]
        problem_string = string_stats[string_stats[:, 0] == problem]
        by_selector = {}
        for selector in selectors:
            selector_int = problem_int[problem_string[:, 1] == selector]
            selector_float = problem_float[problem_string[:, 1] == selector]
            selector_string = problem_string[problem_string[:, 1] == selector]
            by_selector[selector] = {'int': selector_int, 'float': selector_float, 'string': selector_string}
        split[problem] = by_selector

    # Compute and record the 1-shifted geometric mean of solving time, and record the wins and node counts.
    with open(os.path.join(out_dir, "benchmark_stats.csv"), 'w') as file:
        for problem in problems:
            # Print headers.
            headers = ["Time", "Wins", "Nodes"]
            print(problem, *headers, sep=',', file=file)

            time_means = np.zeros(len(selectors))
            time_diffs = np.zeros(len(selectors))
            solve_times = np.zeros((len(split[problem]['hybrid']['string']), len(selectors)))
            node_means = np.zeros(len(selectors), dtype=int)
            node_diffs = np.zeros(len(selectors))
            for i in range(len(selectors)):
                selector = selectors[i]
                int_data = split[problem][selector]['int']
                float_data = split[problem][selector]['float']

                # Compute 1-shifted geometric mean of solving time.
                solve_time = float_data[:, 0]
                k = len(solve_time)
                s = 1
                time_means[i] = np.power(np.prod(np.maximum(solve_time + s, 1)), 1 / k) - s
                time_diffs[i] = 100 * np.sqrt(np.mean(np.power(solve_time - time_means[i], 2))) / time_means[i]

                # Record solve times sorted by instance and seed (in that order).
                sorted_float = float_data[np.lexsort((int_data[:, 1], int_data[:, 0]))]
                solve_times[:, i] = sorted_float[:, 0]

                # Compute the mean number of nodes.
                node_counts = int_data[:, 2]
                node_means[i] = np.round(np.mean(node_counts)).astype(int)
                node_diffs[i] = 100 * np.round(np.std(node_counts)) / node_means[i]

            # Compute the number of wins for each selector.
            baseline_wins = solve_times[:, 0] <= solve_times[:, 1]
            gcnn_wins = 1 - baseline_wins
            baseline_wins = np.sum(baseline_wins)
            gcnn_wins = np.sum(gcnn_wins)
            print("hybrid", f"{time_means[0]:.2f} $\\pm$ {time_diffs[0]:.1f}%", f"{baseline_wins}",
                  f"{node_means[0]:d} $\\pm$ {node_diffs[0]:.1f}%", sep=',', file=file)
            print("gcnn", f"{time_means[1]:.2f} $\\pm$ {time_diffs[1]:.1f}%", f"{gcnn_wins}",
                  f"{node_means[1]:d} $\\pm$ {node_diffs[1]:.1f}%", sep=',', file=file)
            print("", file=file)

        # Print excluded results (if any).
        if len(excluded_rows) > 0:
            print("", file=file)
            print("Excluded", file=file)
            print("Row", "Problem", "Instance", "Seed", sep=',', file=file)
            for i in range(len(excluded_rows)):
                print(excluded_rows[i], *excluded[i], sep=',', file=file)


if __name__ == '__main__':
    # For command line use.
    summarize_stats()
