"""This module provides methods for benchmarking the performance of our GCNN approach.

Summary
=======
This module provides methods for benchmarking the performance of our GCNN approach, compared to a hybrid cut selector
as implemented in SCIP. Benchmarking is performed with respect to the MIPLIB 2010 benchmark test set (or any other
downloaded set, as long as it's placed under data/benchmarking).

Classes
========
- :class:`CustomCutsel`: The hybrid and graph convolutional neural network (GCNN) cut selectors.

Functions
=========
- :func:`benchmark_models`: Benchmark the models in accordance with our benchmarking scheme and write the results to
    a CSV file.
- :func:`process_tasks`: Worker loop: fetch a task and evaluate the given model on the given benchmark instance.
"""

import csv
import glob
import os
from argparse import ArgumentParser
from datetime import timedelta
from math import ceil
from multiprocessing import Manager, Process, Queue, cpu_count
from time import perf_counter, process_time

import numpy as np
import pyscipopt as scip
import tensorflow as tf
from numpy.random import default_rng
from pyscipopt import SCIP_RESULT
from pyscipopt.scip import Cutsel

from model import GCNN
from utils import get_state, init_scip, load_seeds


class CustomCutsel(Cutsel):
    """The hybrid and graph convolutional neural network (GCNN) cut selectors.

    This class extends PySCIPOpt's Cutsel class for user-defined cut selector plugins. The selector implements both
    SCIP's hybrid cut selector and our GCNN approach that ranks the cuts based on estimated bound improvement. One of
    these strategies is used for ranking the cut candidates, depending on whether a GCNN is provided, after which
    they are filtered using parallelism.

    Methods
    =======
    - :meth:`cutselselect`: Select cuts based on estimated bound improvement.

    :ivar function: An optional call function of a trained GCNN model, if provided, this will be used to rank the cut
        candidates. Otherwise, the hybrid approach will be used.
    :ivar p_max: The maximum parallelism for low-quality cuts.
    :ivar p_max_ub: The maximum parallelism for high-quality cuts.
    :ivar skip_factor: The factor that determines the high-quality threshold relative to the highest-quality cut.
    """

    def __init__(self, function=None, p_max=0.1, p_max_ub=0.5, skip_factor=0.9):
        self.use_gcnn = False
        if function is not None:
            self.get_improvements = function
            self.use_gcnn = True

        self.p_max = p_max
        self.p_max_ub = p_max_ub
        self.skip_factor = skip_factor

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """Select cuts based on estimated bound improvement.

        This method is called whenever cuts need to be ranked.

        :param cuts: A list of cut candidates.
        :param forcedcuts: A list of forced cuts, that are not subject to selection.
        :param root: True if we are at the root node (not used).
        :param maxnselectedcuts: The maximum number of selected cuts.
        :return: A dictionary of the form {'cuts': np.array, 'nselectedcuts': int, 'result': SCIP_RESULT},
            where 'cuts' represent the resorted array of cuts in descending order of cut quality, 'nselectedcuts'
            represents the number of cuts that should be selected from cuts (the first 'nselectedcuts'), and 'result'
            signals to SCIP that everything worked out.
        """

        if self.use_gcnn:
            # Extract the state.
            state = get_state(self.model, cuts)
            cons_feats, cons_edge_feats, var_feats, cut_feats, cut_edge_feats = state

            # Convert everything to tensors.
            cons_feats = tf.convert_to_tensor(cons_feats['values'], dtype=tf.float32)
            cons_edge_inds = tf.convert_to_tensor(cons_edge_feats['indices'], dtype=tf.int32)
            cons_edge_feats = tf.convert_to_tensor(cons_edge_feats['values'], dtype=tf.float32)
            var_feats = tf.convert_to_tensor(var_feats['values'], dtype=tf.float32)
            cut_feats = tf.convert_to_tensor(cut_feats['values'], dtype=tf.float32)
            cut_edge_inds = tf.convert_to_tensor(cut_edge_feats['indices'], dtype=tf.int32)
            cut_edge_feats = tf.convert_to_tensor(cut_edge_feats['values'], dtype=tf.float32)
            n_cons = tf.convert_to_tensor(cons_feats.shape[0], dtype=tf.int32)
            n_vars = tf.convert_to_tensor(var_feats.shape[0], dtype=tf.int32)
            n_cuts = tf.convert_to_tensor(cut_feats.shape[0], dtype=tf.int32)

            state = cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, cut_edge_feats,\
                    n_cons, n_vars, n_cuts

            # Get the predicted bound improvements.
            quality = self.get_improvements(state, tf.convert_to_tensor(False, dtype=tf.bool)).numpy()
        else:
            # Use a hybrid cut selection rule.
            if self.model.getBestSol() is not None:
                quality = np.array([self.model.getCutEfficacy(cut) + 0.1 * self.model.getRowNumIntCols(
                    cut) / cut.getNNonz() + 0.1 * self.model.getRowObjParallelism(
                    cut) + 0.5 * self.model.getCutLPSolCutoffDistance(cut, self.model.getBestSol()) for cut in cuts])
            else:
                quality = np.array([1.5 * self.model.getCutEfficacy(cut) + 0.1 * self.model.getRowNumIntCols(
                    cut) / cut.getNNonz() + 0.1 * self.model.getRowObjParallelism(cut) for cut in cuts])

        # Rank the cuts in descending order of quality.
        rankings = sorted(range(len(cuts)), key=lambda x: quality[x], reverse=True)
        sorted_cuts = np.array([cuts[rank] for rank in rankings])

        # Sort cut quality in descending order as well to match the array with sorted cuts.
        quality = -np.sort(-quality)

        # First check whether any cuts are parallel to forced cuts.
        n_selected = len(cuts)
        for cut in forcedcuts:
            # Mark all cuts that are parallel to forced cut i.
            parallelism = [self.model.getRowParallelism(cut, sorted_cuts[j]) for j in range(n_selected)]
            parallelism = np.pad(parallelism, (0, len(cuts) - n_selected), constant_values=0)
            marked = (parallelism > self.p_max)

            # Only remove low-quality or very parallel cuts.
            low_quality = np.logical_or(quality < 0.9 * quality[0], parallelism > self.p_max_ub)
            to_remove = np.logical_and(marked, low_quality)

            # Move cuts that are marked for removal to the back and decrease number of selected cuts.
            removed = sorted_cuts[to_remove]
            sorted_cuts = np.delete(sorted_cuts, to_remove)
            sorted_cuts = np.concatenate((sorted_cuts, removed))
            n_selected -= removed.size

        # Now remove cuts of low quality that are parallel to a cut of higher quality.
        i = 0
        while i < n_selected - 1:
            # Mark all cuts that are parallel to higher-quality cut i.
            parallelism = [self.model.getRowParallelism(sorted_cuts[i], sorted_cuts[j]) for j in
                           range(i + 1, len(sorted_cuts))]
            parallelism = np.pad(parallelism, (i + 1, 0), constant_values=0)
            marked = (parallelism > self.p_max)

            # Only remove low-quality or very parallel cuts.
            low_quality = np.logical_or(quality < 0.9 * quality[0], parallelism > self.p_max_ub)
            to_remove = np.logical_and(marked, low_quality)

            # Move cuts that are marked for removal to the back and decrease number of selected cuts.
            removed = sorted_cuts[to_remove]
            sorted_cuts = np.delete(sorted_cuts, to_remove)
            sorted_cuts = np.concatenate((sorted_cuts, removed))
            n_selected -= removed.size
            i += 1

        return {'cuts': sorted_cuts, 'nselectedcuts': min(n_selected, maxnselectedcuts), 'result': SCIP_RESULT.SUCCESS}


def benchmark_models(n_jobs: int):
    """Benchmark the models in accordance with our benchmarking scheme and write the results to a CSV file.

    The benchmarking is parallelized over multiple cores. Tasks are first sent to a queue, and then processed by the
    workers.

    The CSV file contains the following information:

    - problem: The problem type that was used to train the model, one of {'setcov', 'combauc', 'capfac', 'indset'}.
    - selector: The cut selector used, either 'hybrid' or 'gcnn'.
    - seed: The seed value that was used to train the model.
    - instance: The name of the benchmarking instance considered.
    - n_nodes: The number of nodes processed during solving.
    - n_lps: The number of LPs solved during solving.
    - solve_time: The solving time in CPU seconds.
    - gap: The integrality gap, i.e., :math:`|(primal\\_bound - dual\\_bound)/\\min(|primal\\_bound|,|dual\\_bound|)|`.
    - status: The solution status, 'optimal' if the problem was solved to optimality, 'timelimit' if the solving time
        exceeded one hour. Other status messages may be returned, but these should be disregarded in evaluation as
        solving times are not a fair representation of the performance in these cases (e.g., infeasible).
    - wall_time: The elapsed wall clock time during solving.
    - process_time: The elapsed process time during solving, i.e., the sum of system and user CPU time of the current
    process.

    :param n_jobs: The number of jobs to run in parallel.
    """

    # Start timers.
    wall_start = perf_counter()
    proc_start = process_time()

    print("Benchmarking models...")

    # Load the (MIPLIB 2010) benchmark set.
    instances = glob.glob(f"data/benchmarking/*.mps")

    os.makedirs('results', exist_ok=True)

    problems = ['setcov', 'combauc', 'capfac', 'indset']
    cut_selectors = ['hybrid', 'gcnn']
    seeds = load_seeds(name='train_seeds')

    # Disable GPU for fair measurement.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')

    # Schedule jobs.
    manager = Manager()
    task_queue = manager.Queue()
    for problem in problems:
        for selector in cut_selectors:
            for seed in seeds:
                for instance in instances:
                    name, _ = os.path.splitext(os.path.basename(instance))
                    task_queue.put((problem, selector, seed, name, instance))

    # Append worker termination signals to the queue.
    for i in range(n_jobs):
        task_queue.put('done')

    # Start workers and tell them to process orders from the task queue.
    out_queue = manager.Queue()
    workers = []
    for i in range(n_jobs):
        worker = Process(target=process_tasks, args=(task_queue, out_queue), daemon=True)
        workers.append(worker)
        worker.start()

    # Wait for all workers to finish.
    for worker in workers:
        worker.join()

    # Append an end message to the queue.
    out_queue.put('end')

    # Write the results to a CSV file.
    fieldnames = ['problem', 'selector', 'seed', 'instance', 'n_nodes', 'n_lps', 'solve_time', 'gap', 'status',
                  'wall_time', 'process_time']
    with open('results/benchmark.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            # Fetch a result from the queue.
            result = out_queue.get()
            if result == 'end':
                # We have reached the end of the queue.
                break

            (problem, selector, seed, instance, n_nodes, n_lps, solve_time, gap, status, wall_time, proc_time) = result
            writer.writerow(
                {'problem': problem, 'selector': selector, 'seed': seed, 'instance': instance, 'n_nodes': n_nodes,
                 'n_lps': n_lps, 'solve_time': solve_time, 'gap': gap, 'status': status, 'wall_time': wall_time,
                 'process_time': proc_time})
    print("Done!")
    print(f"Wall time: {str(timedelta(seconds=ceil(perf_counter() - wall_start)))}")
    print(f"CPU time: {str(timedelta(seconds=ceil(process_time() - proc_start)))}")


def process_tasks(task_queue: Queue, out_queue: Queue):
    """Worker loop: fetch a task and evaluate the given model on the given benchmark instance.

    :param task_queue: The task queue from which the worker needs to fetch tasks.
    :param out_queue: The out queue to which the worker should send results.
    """

    while True:
        # Fetch a task.
        task = task_queue.get()

        if task == 'done':
            # This worker is done.
            break

        problem, selector, seed, instance, path = task

        rng = np.random.default_rng(seed)
        tf.random.set_seed(rng.integers(np.iinfo(int).max))
        scip_seed = rng.integers(2147483648)

        # Initialize model.
        model = scip.Model()
        init_scip(model, scip_seed, cpu_time=True)
        model.readProblem(path)

        if selector == 'hybrid':
            # Solve using the hybrid cut selector.
            cut_selector = CustomCutsel()
            model.includeCutsel(cut_selector, 'hybrid cut selector',
                                'selects cuts based on weighted average of quality metrics', 5000000)
        else:
            # Solve using our GCNN cut selector.
            gcnn = GCNN()

            # Load the trained model.
            gcnn.restore_state(f'trained_models/{problem}/{seed}/best_params.pkl')

            # Precompile the forward pass (call) for performance.
            gcnn.call = tf.function(gcnn.call, input_signature=gcnn.input_signature)
            get_improvements = gcnn.call.get_concrete_function()

            cut_selector = CustomCutsel(function=get_improvements)
            model.includeCutsel(cut_selector, 'GCNN cut selector', 'selects cuts based on estimated bound improvement',
                                5000000)

        # Start timers.
        wall_time = perf_counter()
        proc_time = process_time()

        # Optimize the problem.
        model.optimize()

        # Record times.
        wall_time = perf_counter() - wall_time
        proc_time = process_time() - proc_time

        # Record SCIP statistics.
        solve_time = model.getSolvingTime()
        n_nodes = model.getNNodes()
        n_lps = model.getNLPs()
        gap = model.getGap()
        status = model.getStatus()

        # Send result to the out queue.
        out_queue.put(
            (problem, selector, seed, instance, n_nodes, n_lps, solve_time, gap, status, wall_time, proc_time))
        model.freeProb()


if __name__ == '__main__':
    # For command line use.
    parser = ArgumentParser()
    parser.add_argument('-j', '--n_jobs', help='The number of jobs to run in parallel (default: all cores).',
                        default=cpu_count())
    args = parser.parse_args()

    print(f"Running {args.n_jobs} jobs in parallel.")
    benchmark_models(args.n_jobs)
