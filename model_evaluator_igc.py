"""This module provides methods for evaluating the performance of our GCNN approach in terms of integrality gap
closure (IGC).

Summary
=======
This module provides methods for evaluating the performance of our GCNN approach in terms of IGC, compared to a
hybrid cut selector as implemented in SCIP.

Classes
========
- :class:`CustomCutsel`: The hybrid and graph convolutional neural network (GCNN) cut selectors.

Functions
=========
- :func:`benchmark_models`: Evaluate the models in accordance with our evaluation scheme for IGC and write the
    results to a CSV file.
- :func:`process_tasks`: Worker loop: fetch a task and evaluate the given model on the given IGC evaluation instance.
"""

import csv
import glob
import os
from argparse import ArgumentParser
from datetime import timedelta
from math import ceil
from multiprocessing import Manager, Process, Queue, cpu_count
from resource import RUSAGE_CHILDREN, RUSAGE_SELF, getrusage
from time import perf_counter

import numpy as np
import pyscipopt as scip
import tensorflow as tf
from numpy.random import default_rng
from pyscipopt import SCIP_LPSOLSTAT, SCIP_RESULT
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

    def __init__(self, sol: float, problem: str, out_queue: Queue, function=None, p_max=0.1, p_max_ub=0.5,
                 skip_factor=0.9):
        self.use_gcnn = False
        if function is not None:
            self.get_improvements = function
            self.use_gcnn = True

        self.p_max = p_max
        self.p_max_ub = p_max_ub
        self.skip_factor = skip_factor

        self.sol = sol
        self.problem = problem
        self.out_queue = out_queue
        self.z_0 = 0

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

        if self.model.getNSepaRounds() == 0:
            self.z_0 = self.model.getLPObjVal()

        if self.model.getNNodes() == 1 and self.model.getNSepaRounds() < 30:
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

                state = cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, \
                        cut_edge_feats, n_cons, n_vars, n_cuts

                # Get the predicted bound improvements.
                quality = self.get_improvements(state, tf.convert_to_tensor(False, dtype=tf.bool)).numpy()
            else:
                # Use a hybrid cut selection rule.
                quality = np.array([self.model.getCutEfficacy(cut) + 0.1 * self.model.getRowNumIntCols(
                    cut) / cut.getNNonz() + 0.1 * self.model.getRowObjParallelism(cut) for cut in cuts])

            # Rank the cuts in descending order of quality.
            rankings = sorted(range(len(cuts)), key=lambda x: quality[x], reverse=True)
            sorted_cuts = np.array([cuts[rank] for rank in rankings])

            # Add the cut to the LP relaxation and solve it.
            self.model.startDive()
            self.model.addRowDive(sorted_cuts[0])
            self.model.constructLP()
            self.model.solveDiveLP()

            # Make sure that the LP bound is feasible.
            quality = -np.Inf
            solstat = self.model.getLPSolstat()

            # Do not record this result if SCIP couldn't solve the LP to optimality.
            if solstat == SCIP_LPSOLSTAT.OPTIMAL:
                z_k = self.model.getLPObjVal()

                # Compute the IGC.
                igc = (z_k - self.z_0) / (self.sol - self.z_0)

                selector = 'gcnn' if self.use_gcnn else 'hybrid'
                self.out_queue.put((self.problem, selector, self.model.getNSepaRounds() + 1, igc))
            self.model.endDive()
        else:
            # Use a hybrid cut selection rule.
            quality = np.array([self.model.getCutEfficacy(cut) + 0.1 * self.model.getRowNumIntCols(
                cut) / cut.getNNonz() + 0.1 * self.model.getRowObjParallelism(cut) for cut in cuts])

            # Rank the cuts in descending order of quality.
            rankings = sorted(range(len(cuts)), key=lambda x: quality[x], reverse=True)
            sorted_cuts = np.array([cuts[rank] for rank in rankings])

        return {'cuts': sorted_cuts, 'nselectedcuts': 1, 'result': SCIP_RESULT.SUCCESS}


def evaluate_models_igc(n_jobs: int):
    """Evaluate the models in accordance with our evaluation scheme for IGC and write the results to a CSV file.

    The evaluation is parallelized over multiple cores. Tasks are first sent to a queue, and then processed by the
    workers.

    The CSV file contains the following information:

    - problem: The problem type that was used to train the model, one of {'setcov', 'combauc', 'capfac', 'indset'}.
    - selector: The cut selector used, either 'hybrid' or 'gcnn'.
    - cut: The number of this cut, between 1-30.
    - igc: The integrality gap closure.

    :param n_jobs: The number of jobs to run in parallel.
    """

    # Start timer.
    wall_start = perf_counter()

    print("Evaluating models in terms of IGC...")

    os.makedirs('results', exist_ok=True)

    problems = ['setcov', 'combauc', 'capfac', 'indset']
    dims = {'setcov': '500r', 'combauc': '100i_500b', 'capfac': '100c_100f', 'indset': '500n'}
    seeds = load_seeds(name='train_seeds')

    # Disable GPU for fair measurement.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')

    # Schedule jobs.
    manager = Manager()
    task_queue = manager.Queue()
    for problem in problems:
        # Load the (MIPLIB 2010) benchmark set.
        instances = sorted(glob.glob(f"data/instances/{problem}/eval_igc_{dims[problem]}/*.lp"))

        # Find the best model.
        best_loss = np.Inf
        best_seed = 0
        for seed in seeds:
            loss = np.load(f"results/test/{problem}/{seed}_loss.npy")
            if loss < best_loss:
                best_loss = loss
                best_seed = seed
        for instance in instances:
            task_queue.put((problem, instance, best_seed))

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
    fieldnames = ['problem', 'selector', 'cut', 'igc']
    with open('results/eval_igc.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            # Fetch a result from the queue.
            result = out_queue.get()
            if result == 'end':
                # We have reached the end of the queue.
                break

            problem, selector, cut, igc = result
            writer.writerow({'problem': problem, 'selector': selector, 'cut': cut, 'igc': igc})

    # Get combined usage of all processes.
    usage_self = getrusage(RUSAGE_SELF)
    usage_children = getrusage(RUSAGE_CHILDREN)
    cpu_time = usage_self[0] + usage_self[1] + usage_children[0] + usage_children[1]

    print("Done!")
    print(f"Wall time: {str(timedelta(seconds=ceil(perf_counter() - wall_start)))}")
    print(f"CPU time: {str(timedelta(seconds=ceil(cpu_time)))}")


def process_tasks(task_queue: Queue, out_queue: Queue):
    """Worker loop: fetch a task and evaluate the given model on the given IGC evaluation instance.

    :param task_queue: The task queue from which the worker needs to fetch tasks.
    :param out_queue: The out queue to which the worker should send results.
    """

    while True:
        # Fetch a task.
        task = task_queue.get()

        if task == 'done':
            # This worker is done.
            break

        problem, instance, seed = task

        rng = np.random.default_rng(123)
        tf.random.set_seed(int(rng.integers(np.iinfo(int).max)))
        scip_seed = rng.integers(2147483648)

        # Initialize model for initial solution.
        model = scip.Model()
        init_scip(model, scip_seed, most_inf=False, cpu_time=True)
        model.readProblem(instance)

        model.optimize()
        sol = model.getObjVal()
        model.freeProb()

        # Initialize model for hybrid cut selector.
        model = scip.Model()
        init_scip(model, scip_seed, most_inf=False, cpu_time=True)
        model.setParam("limits/nodes", 1)
        model.readProblem(instance)

        # Solve using the hybrid cut selector.
        cut_selector = CustomCutsel(sol, problem, out_queue)
        model.includeCutsel(cut_selector, 'hybrid cut selector',
                            'selects cuts based on weighted average of quality metrics', 5000000)

        # Optimize the problem.
        model.optimize()
        model.freeProb()

        # Initialize model for GCNN cut selector.
        model = scip.Model()
        init_scip(model, scip_seed, most_inf=False, cpu_time=True)
        model.setParam("limits/nodes", 1)
        model.readProblem(instance)

        # Solve using our GCNN cut selector.
        gcnn = GCNN()

        # Load the trained model.
        gcnn.restore_state(f'trained_models/{problem}/{seed}/best_params.pkl')

        # Precompile the forward pass (call) for performance.
        gcnn.call = tf.function(gcnn.call, input_signature=gcnn.input_signature)
        get_improvements = gcnn.call.get_concrete_function()

        cut_selector = CustomCutsel(sol, problem, out_queue, function=get_improvements)
        model.includeCutsel(cut_selector, 'GCNN cut selector', 'selects cuts based on estimated bound improvement',
                            5000000)

        # Optimize the problem.
        model.optimize()
        model.freeProb()


if __name__ == '__main__':
    # For command line use.
    parser = ArgumentParser()
    parser.add_argument('-j', '--n_jobs', help='The number of jobs to run in parallel (default: all cores).',
                        default=cpu_count())
    args = parser.parse_args()

    print(f"Running {args.n_jobs} jobs in parallel.")
    evaluate_models_igc(args.n_jobs)
