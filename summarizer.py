import os

import numpy as np

from utils import load_seeds


def summarize_stats():
    out_dir = "summaries"
    os.makedirs(out_dir)

    summarize_sampling(out_dir)
    summarize_testing(out_dir)
    summarize_evaluation(out_dir)


def summarize_sampling(out_dir: str):
    dims = {'setcov': '50r', 'combauc': '10i_50b', 'capfac': '10c', 'indset': '50n'}

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
                gcnn_fracs[i, :] = stats[0, 2:]
                baseline_fracs[i, :] = stats[1, 2:]
            gcnn_means = 100 * np.mean(gcnn_fracs, axis=0)
            gcnn_sds = np.std(gcnn_fracs, axis=0)
            baseline_means = 100 * np.mean(baseline_fracs, axis=0)
            baseline_sds = np.std(baseline_fracs, axis=0)
            print("gcnn", f"{gcnn_means[0]:.2f} ({gcnn_sds[0]:.2f})", f"{gcnn_means[1]:.2f} ({gcnn_sds[1]:.2f})",
                  f"{gcnn_means[2]:.2f} ({gcnn_sds[2]:.2f})", f"{gcnn_means[3]:.2f} ({gcnn_sds[3]:.2f})", sep=',',
                  file=file)
            print("baseline", f"{baseline_means[0]:.2f} ({baseline_sds[0]:.2f})",
                  f"{baseline_means[1]:.2f} ({baseline_sds[1]:.2f})",
                  f"{baseline_means[2]:.2f} ({baseline_sds[2]:.2f})",
                  f"{baseline_means[3]:.2f} ({baseline_sds[3]:.2f})", sep=',', file=file)
            print("", file=file)


def summarize_evaluation(out_dir: str):
    # Load the evaluation file (needs to be split up as NumPy arrays cannot have multiple dtypes).
    # [problem, difficulty, selector, status].
    string_stats = np.genfromtxt("results/eval.csv", dtype=str, delimiter=',', skip_header=1, usecols=(0, 1, 3, 9))

    # [instance, seed, n_nodes, n_lps].
    int_stats = np.genfromtxt("results/eval.csv", dtype=int, delimiter=',', skip_header=1, usecols=(2, 4, 5, 6))

    # [solve_time, gap, wall_time, process_time].
    float_stats = np.genfromtxt("results/eval.csv", dtype=float, delimiter=',', skip_header=1, usecols=(7, 8, 10, 11))

    # Filter stats.
    excluded = np.logical_not(np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit'))
    excluded = np.arange(2, len(string_stats) + 2) * excluded  # Get row numbers in eval.csv.
    excluded = excluded[excluded != 0]
    int_stats = int_stats[np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit')]
    float_stats = float_stats[np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit')]
    string_stats = string_stats[np.logical_or(string_stats[:, 3] == 'optimal', string_stats[:, 3] == 'timelimit')]

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
                time_sds = np.zeros(len(selectors))
                solve_times = np.zeros((len(split[problem][difficulty]['hybrid']['string']), len(selectors)))
                node_means = np.zeros(len(selectors), dtype=int)
                node_sds = np.zeros(len(selectors), dtype=int)
                for i in range(len(selectors)):
                    selector = selectors[i]
                    int_data = split[problem][difficulty][selector]['int']
                    float_data = split[problem][difficulty][selector]['float']

                    # Compute 1-shifted geometric mean of solving time.
                    solve_time = float_data[:, 0]
                    k = len(solve_time)
                    s = 1
                    time_means[i] = np.power(np.prod(np.maximum(solve_time + s, 1)), 1 / k) - s
                    time_sds[i] = np.sqrt(np.mean(np.power(solve_time - time_means[i], 2)))

                    # Record solve times sorted by instance and seed (in that order).
                    sorted_float = float_data[np.lexsort((int_data[:, 1], int_data[:, 0]))]
                    solve_times[:, i] = sorted_float[:, 0]

                    # Compute the mean number of nodes.
                    node_counts = int_data[:, 2]
                    node_means[i] = np.round(np.mean(node_counts)).astype(int)
                    node_sds[i] = np.round(np.std(node_counts)).astype(int)

                # Compute the number of wins for each selector.
                baseline_wins = solve_times[:, 0] <= solve_times[:, 1]
                gcnn_wins = 1 - baseline_wins
                baseline_wins = np.sum(baseline_wins)
                gcnn_wins = np.sum(gcnn_wins)
                lines[0] += ["", f"{time_means[0]:.2f} ({time_sds[0]:.2f})", f"{baseline_wins} / 100",
                             f"{node_means[0]:d} ({node_sds[0]:d})"]
                lines[1] += ["", f"{time_means[1]:.2f} ({time_sds[1]:.2f})", f"{gcnn_wins} / 100",
                             f"{node_means[1]:d} ({node_sds[1]:d})"]
            for line in lines:
                print(*line, sep=',', file=file)
            print("", file=file)

        # Print excluded results (if any).
        if len(excluded) > 0:
            print("", file=file)
            print("Excluded:", file=file)
            print(*excluded, sep=',', file=file)
