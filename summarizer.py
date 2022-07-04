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
- :func:`perc_std`: Computes the percentual standard deviation over an array-like object.
- :func:`summarize_benchmarking`: Summarize the model benchmarking results.
"""

import os
from datetime import timedelta
from math import ceil
from time import perf_counter, process_time

import numpy as np
import pandas as pd

from utils import load_seeds


def summarize_stats():
    """Summarize all obtained results and writes the summaries to CSV files."""

    # Start timers.
    wall_start = perf_counter()
    proc_start = process_time()

    out_dir = "summaries"
    os.makedirs(out_dir)

    summarize_sampling(out_dir)
    summarize_testing(out_dir)
    summarize_evaluation(out_dir)
    summarize_benchmarking(out_dir)

    print("Done!")
    print(f"Wall time: {str(timedelta(seconds=ceil(perf_counter() - wall_start)))}")
    print(f"CPU time: {str(timedelta(seconds=ceil(process_time() - proc_start)))}")


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
            random_fracs = np.zeros((len(train_seeds), 4))
            hybrid_fracs = np.zeros((len(train_seeds), 4))
            gcnn_fracs = np.zeros((len(train_seeds), 4))
            for i in range(len(train_seeds)):
                stats = np.genfromtxt(files[str(train_seeds[i])], delimiter=',', skip_header=1)
                random_fracs[i, :] = 100 * stats[0, 2:]
                hybrid_fracs[i, :] = 100 * stats[1, 2:]
                gcnn_fracs[i, :] = 100 * stats[2, 2:]
            random_means = np.mean(random_fracs, axis=0)
            random_sds = np.std(random_fracs, axis=0)
            hybrid_means = hybrid_fracs[0, :]  # Hybrid cut selector is deterministic.
            hybrid_sds = np.zeros(4)
            gcnn_means = np.mean(gcnn_fracs, axis=0)
            gcnn_sds = np.std(gcnn_fracs, axis=0)
            print("random", f"${random_means[0]:.2f} \\pm {random_sds[0]:.2f}$",
                  f"${random_means[1]:.2f} \\pm {random_sds[1]:.2f}$",
                  f"${random_means[2]:.2f} \\pm {random_sds[2]:.2f}$",
                  f"${random_means[3]:.2f} \\pm {random_sds[3]:.2f}$", sep=',', file=file)
            print("hybrid", f"${hybrid_means[0]:.2f} \\pm {hybrid_sds[0]:.2f}$",
                  f"${hybrid_means[1]:.2f} \\pm {hybrid_sds[1]:.2f}$",
                  f"${hybrid_means[2]:.2f} \\pm {hybrid_sds[2]:.2f}$",
                  f"${hybrid_means[3]:.2f} \\pm {hybrid_sds[3]:.2f}$", sep=',', file=file)
            print("gcnn", f"${gcnn_means[0]:.2f} \\pm {gcnn_sds[0]:.2f}$",
                  f"${gcnn_means[1]:.2f} \\pm {gcnn_sds[1]:.2f}$", f"${gcnn_means[2]:.2f} \\pm {gcnn_sds[2]:.2f}$",
                  f"${gcnn_means[3]:.2f} \\pm {gcnn_sds[3]:.2f}$", sep=',', file=file)
            print("", file=file)


def summarize_evaluation(out_dir: str):
    """Summarize the model evaluation results.

    :param out_dir: The directory to store the summary in.
    """

    # Load the evaluation file.
    stats = pd.read_csv("results/eval.csv")

    # Filter stats.
    excluded = ~((stats['status'] == 'optimal') | (stats['status'] == 'timelimit')).to_numpy()
    excluded_entries = stats[excluded].loc[:, ['problem', 'difficulty', 'instance', 'seed']]
    excluded_rows = excluded_entries.index + 2  # Get row numbers in eval.csv.
    excluded_entries = excluded_entries.to_numpy()
    stats = stats[~excluded]

    # Compute and record the 1-shifted geometric mean of solving time, and record the wins and node counts.
    problems = ['setcov', 'combauc', 'capfac', 'indset']
    difficulties = ['easy', 'medium', 'hard']
    selectors = ['hybrid', 'gcnn']
    with open(os.path.join(out_dir, "eval_stats.csv"), 'w') as file:
        # Print headers.
        print("", "", "Easy", "", "", "", "Medium", "", "", "", "Hard", "", sep=',', file=file)
        for problem in problems:
            # Print headers.
            headers = 3 * ["Time", "Wins", "Nodes", ""]
            print(problem, *headers, sep=',', file=file)

            lines = [[], []]
            for difficulty in difficulties:
                sub_stats = stats[(stats['problem'] == problem) & (stats['difficulty'] == difficulty)]
                sub_stats.set_index(['instance', 'seed'], inplace=True)

                # Compute the mean solve time and number of wins, considering only instances considered by both.
                hybrid_stats = sub_stats[sub_stats['selector'] == 'hybrid']
                gcnn_stats = sub_stats[sub_stats['selector'] == 'gcnn']
                in_both = hybrid_stats.index.intersection(gcnn_stats.index)
                sub_stats = sub_stats.loc[in_both, :]

                time_means = np.zeros(len(selectors))
                time_diffs = np.zeros(len(selectors))
                solve_times = []
                hybrid_wins = 0
                gcnn_wins = 0
                if len(sub_stats) != 0:
                    for i in range(len(selectors)):
                        # Retrieve stats for this selector and sort it by instance and seed.
                        subsub_stats = sub_stats[sub_stats['selector'] == selectors[i]]

                        # Compute 1-shifted geometric mean of solving time.
                        solve_time = subsub_stats.loc[:, 'solve_time'].to_numpy()
                        k = len(solve_time)
                        s = 1
                        time_means[i] = np.exp(np.sum(np.log(np.maximum(solve_time + s, 1))) / k) - s

                        # Compute the average per-instance standard deviation.
                        time_diff = subsub_stats.loc[:, 'solve_time'].groupby('instance').aggregate(perc_std).to_numpy()
                        time_diffs[i] = np.mean(time_diff)

                        # Record solve times.
                        solve_times.append(solve_time)

                    # Compute the number of wins for each selector.
                    solve_times = np.array(solve_times)
                    hybrid_wins = np.logical_and(solve_times[0, :] < solve_times[1, :], solve_times[0, :] < 3600)
                    gcnn_wins = np.logical_and(solve_times[1, :] < solve_times[0, :], solve_times[1, :] < 3600)
                    hybrid_wins = np.sum(hybrid_wins)
                    gcnn_wins = np.sum(gcnn_wins)

                # Compute the mean number of nodes for each selector, considering only instances solved by both.
                both_solved = (hybrid_stats['status'] == 'optimal') & (gcnn_stats['status'] == 'optimal')
                sub_stats = sub_stats.loc[both_solved, :]

                node_means = np.zeros(len(selectors), dtype=int)
                node_diffs = np.zeros(len(selectors))
                if len(sub_stats) != 0:
                    for i in range(len(selectors)):
                        subsub_stats = sub_stats[sub_stats['selector'] == selectors[i]]

                        # Compute 1-shifted geometric mean of node counts.
                        node_counts = subsub_stats.loc[:, 'n_nodes'].to_numpy()
                        k = len(node_counts)
                        s = 1
                        node_means[i] = np.round(np.exp(np.sum(np.log(np.maximum(node_counts + s, 1))) / k) - s)

                        # Compute the average per-instance standard deviation.
                        node_diff = subsub_stats.loc[:, 'n_nodes'].groupby('instance').aggregate(perc_std).to_numpy()
                        node_diffs[i] = np.mean(node_diff)

                lines[0] += [f"${time_means[0]:.2f} \\pm {time_diffs[0]:.1f}\\%$", f"{hybrid_wins}",
                             f"${node_means[0]:d} \\pm {node_diffs[0]:.1f}\\%$", ""]
                lines[1] += [f"${time_means[1]:.2f} \\pm {time_diffs[1]:.1f}\\%$", f"{gcnn_wins}",
                             f"${node_means[1]:d} \\pm {node_diffs[1]:.1f}\\%$", ""]
            print("hybrid", *lines[0], sep=',', file=file)
            print("gcnn", *lines[1], sep=',', file=file)
            print("", file=file)

        # Print excluded results (if any).
        if len(excluded_rows) > 0:
            print("", file=file)
            print("Excluded", file=file)
            print("Row", "Problem", "Difficulty", "Instance", "Seed", sep=',', file=file)
            for i in range(len(excluded_rows)):
                print(excluded_rows[i], *excluded_entries[i], sep=',', file=file)


def perc_std(x):
    """Computes the percentual standard deviation over an array-like object.

    :param x: An array-like object.
    :return: The percentual standard deviation.
    """

    return 100 * np.std(x) / np.mean(x)


def summarize_benchmarking(out_dir: str):
    """Summarize the benchmarking results.

    :param out_dir: The directory to store the summary in.
    """

    # Load the benchmarking file.
    stats = pd.read_csv("results/benchmark.csv")

    # Filter stats.
    excluded = ~((stats['status'] == 'optimal') | (stats['status'] == 'timelimit')).to_numpy()
    excluded_entries = stats[excluded].loc[:, ['problem', 'instance', 'seed']]
    excluded_rows = excluded_entries.index + 2  # Get row numbers in benchmark.csv.
    excluded_entries = excluded_entries.to_numpy()
    stats = stats[~excluded]

    problems = ['setcov', 'combauc', 'capfac', 'indset']
    selectors = ['hybrid', 'gcnn']
    with open(os.path.join(out_dir, "benchmark_stats.csv"), 'w') as file:
        for problem in problems:
            # Print headers.
            headers = ["Time", "Wins", "Nodes"]
            print(problem, *headers, sep=',', file=file)

            sub_stats = stats[stats['problem'] == problem]
            sub_stats.set_index('instance', inplace=True)

            # Compute the mean solve time and number of wins, considering only instances considered by both.
            hybrid_stats = sub_stats[sub_stats['selector'] == 'hybrid']
            gcnn_stats = sub_stats[sub_stats['selector'] == 'gcnn']
            in_both = hybrid_stats.index.intersection(gcnn_stats.index)
            sub_stats = sub_stats.loc[in_both, :]

            time_means = np.zeros(len(selectors))
            solve_times = []
            hybrid_wins = 0
            gcnn_wins = 0
            if len(sub_stats) != 0:
                for i in range(len(selectors)):
                    # Retrieve stats for this selector and sort it by instance and seed.
                    subsub_stats = sub_stats[sub_stats['selector'] == selectors[i]]

                    # Compute 1-shifted geometric mean of solving time.
                    solve_time = subsub_stats.loc[:, 'solve_time'].to_numpy()
                    k = len(solve_time)
                    s = 1
                    time_means[i] = np.exp(np.sum(np.log(np.maximum(solve_time + s, 1))) / k) - s

                    # Record solve times.
                    solve_times.append(solve_time)

                # Compute the number of wins for each selector.
                solve_times = np.array(solve_times)
                hybrid_wins = np.logical_and(solve_times[0, :] < solve_times[1, :], solve_times[0, :] < 3600)
                gcnn_wins = np.logical_and(solve_times[1, :] < solve_times[0, :], solve_times[1, :] < 3600)
                hybrid_wins = np.sum(hybrid_wins)
                gcnn_wins = np.sum(gcnn_wins)

            # Compute the mean number of nodes for each selector, considering only instances solved by both.
            both_solved = (hybrid_stats['status'] == 'optimal') & (gcnn_stats['status'] == 'optimal')
            sub_stats = sub_stats.loc[both_solved, :]

            node_means = np.zeros(len(selectors), dtype=int)
            if len(sub_stats) != 0:
                for i in range(len(selectors)):
                    subsub_stats = sub_stats[sub_stats['selector'] == selectors[i]]

                    # Compute 1-shifted geometric mean of node counts.
                    node_counts = subsub_stats.loc[:, 'n_nodes'].to_numpy()
                    k = len(node_counts)
                    s = 1
                    node_means[i] = np.round(np.exp(np.sum(np.log(np.maximum(node_counts + s, 1))) / k) - s)
            print("hybrid", f"{time_means[0]:.2f}", f"{hybrid_wins}", f"{node_means[0]:d}", sep=',', file=file)
            print("gcnn", f"{time_means[1]:.2f}", f"{gcnn_wins}", f"{node_means[1]:d}", sep=',', file=file)
            print("", file=file)

        # Print excluded results (if any).
        if len(excluded_rows) > 0:
            print("", file=file)
            print("Excluded", file=file)
            print("Row", "Problem", "Instance", "Seed", sep=',', file=file)
            for i in range(len(excluded_rows)):
                print(excluded_rows[i], *excluded_entries[i], sep=',', file=file)


if __name__ == '__main__':
    # For command line use.
    summarize_stats()
