"""This module provides methods for evaluating the performance of our GCNN approach.

Summary
=======
This module provides methods for evaluating the performance of our GCNN approach, compared to a hybrid cut selector
as implemented in SCIP. The methods in this module are based the code by [1]_.

Classes
========
- :class:`CustomCutsel`: The hybrid and graph convolutional neural network (GCNN) cut selectors.

Functions
=========
- :func:`evaluate_models`: Evaluates the models in accordance with our evaluation scheme.
- :func:`evaluate_problem`: Evaluates trained models on a given problem type and writes the results to a CSV file.

References
==========
.. [1] Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). Exact combinatorial optimization with
    graph convolutional neural networks. *Neural Information Processing Systems (NeurIPS 2019)*, 15580–15592.
    https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
"""

import csv
import os
import time

import numpy as np
import pyscipopt as scip
import tensorflow as tf
from numpy.random import default_rng
from pyscipopt import SCIP_RESULT
from pyscipopt.scip import Cutsel

from model import GCNN
from utils import get_state, init_scip


class CustomCutsel(Cutsel):
    """The hybrid and graph convolutional neural network (GCNN) cut selectors.

    This class extends PySCIPOpt's Cutsel class for user-defined cut selector plugins. The selector implements both
    SCIP's hybrid cut selector and our GCNN approach that ranks the cuts based on estimated bound improvement. One of
    these strategies is used for ranking the cut candidates, depending on whether a GCNN is provided, after which
    they are filtered using parallelism.

    Methods
    =======
    - :meth:`cutselselect`: This method is called whenever cuts need to be ranked.

    :ivar gcnn: An optional trained GCNN model, if provided, this will be used to rank the cut candidates. Otherwise,
        the hybrid approach will be used.
    :ivar p_expert: The probability of querying the expert on each cut selection round.
    :ivar p_max: The maximum parallelism for low-quality cuts.
    :ivar p_max_ub: The maximum parallelism for high-quality cuts.
    :ivar skip_factor: The factor that determines the high-quality threshold relative to the highest-quality cut.
    """

    def __init__(self, gcnn=None, p_expert=0.05, p_max=0.1, p_max_ub=0.5, skip_factor=0.9):
        self.use_gcnn = False
        if isinstance(gcnn, GCNN):
            # Compile the GCNN model call as TensorFlow function for performance.
            self.get_improvements = tf.function(gcnn.call, input_signature=gcnn.input_signature)
            self.use_gcnn = True

        self.p_expert = p_expert
        self.p_max = p_max
        self.p_max_ub = p_max_ub
        self.skip_factor = skip_factor

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """Selects cuts based on estimated bound improvement.

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
            quality = self.get_improvements(state, tf.convert_to_tensor(False)).numpy()
        else:
            # Use a hybrid cut selection rule.
            quality = [self.model.getCutEfficacy(cut) + 0.1 * self.model.getRowNumIntCols(
                cut) / cut.getNNonz() + 0.1 * self.model.getRowObjParallelism(
                cut) + 0.5 * self.model.getCutLPSolCutoffDistance(cut, self.model.getBestSol()) for cut in cuts]

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


def evaluate_models(seed: int):
    """Evaluates the models in accordance with our evaluation scheme.

    :param seed: The same seed value that was used to train the models.
    """

    seed_generator = default_rng(seed)
    seeds = seed_generator.integers(2 ** 32, size=5)

    print("Evaluating models...")
    evaluate_problem('setcov', seeds)
    evaluate_problem('combauc', seeds)
    evaluate_problem('capfac', seeds)
    evaluate_problem('indset', seeds)
    print("Done!")


def evaluate_problem(problem: str, seeds: np.array):
    """Evaluates trained models on a given problem type and writes the results to a CSV file.

    The CSV file contains the following information:

    - selector: The cut selector used, either 'hybrid' or 'gcnn'.
    - difficulty: The difficulty level of the instance evaluated, one of {'easy', 'medium', 'hard'}.
    - instance: The instance number considered.
    - seed: The seed value that was used to train the model.
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

    :param problem: The problem type to be considered, one of: {'setcov', 'combauc', 'capfac', or 'indset'}.
    :param seeds: A list of seeds that were used for training the models.
    """

    setcov_folders = ['setcov/eval_500r', 'setcov/eval_1000r', 'setcov/eval_2000r']
    combauc_folders = ['combauc/eval_100i_500b', 'combauc/eval_200i_1000b', 'combauc/eval_300i_1500b']
    capfac_folders = ['capfac/eval_100c', 'capfac/eval_200c', 'capfac/eval_400c']
    indset_folders = ['indset/eval_500n', 'indset/eval_1000n', 'indset/eval_1500n']
    folders = {'setcov': setcov_folders, 'combauc': combauc_folders, 'capfac': capfac_folders, 'indset': indset_folders}
    problem_folders = folders[problem]

    os.makedirs('results', exist_ok=True)
    result_file = f"results/{problem}_eval.csv"

    # Retrieve evaluation instances.
    instances = []

    instances += [
        {'difficulty': 'easy', 'instance': i + 1, 'path': f"data/instances/{problem_folders[0]}/instance_{i + 1}.lp"}
        for i in range(20)]
    instances += [
        {'difficulty': 'medium', 'instance': i + 1, 'path': f"data/instances/{problem_folders[1]}/instance_{i + 1}.lp"}
        for i in range(20)]
    instances += [
        {'difficulty': 'hard', 'instance': i + 1, 'path': f"data/instances/{problem_folders[2]}/instance_{i + 1}.lp"}
        for i in range(20)]

    cut_selectors = ['hybrid', 'gcnn']

    fieldnames = ['selector', 'difficulty', 'instance', 'seed', 'n_nodes', 'n_lps', 'solve_time', 'gap', 'status',
                  'wall_time', 'process_time']
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            for selector in cut_selectors:
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    tf.random.set_seed(rng.integers(np.iinfo(int).max))
                    scip_seed = rng.integers(2147483648)

                    # Initialize model.
                    model = scip.Model()
                    init_scip(model, scip_seed, cpu_time=True)
                    model.readProblem(f"{instance['path']}")

                    if selector == 'hybrid':
                        # Solve using the hybrid cut selector.
                        cut_selector = CustomCutsel()
                        model.includeCutsel(cut_selector, 'hybrid cut selector',
                                            'selects cuts based on weighted average of quality metrics', 5000000)
                    else:
                        # Solve using our GCNN cut selector.
                        gcnn = GCNN().restore_state(f'trained_models/{problem}/{seed}/best_params.pkl')
                        cut_selector = CustomCutsel(gcnn=gcnn)
                        model.includeCutsel(cut_selector, 'GCNN cut selector',
                                            'selects cuts based on estimated bound improvement', 5000000)

                    # Start timers.
                    wall_time = time.perf_counter()
                    process_time = time.process_time()

                    # Optimize the problem.
                    model.optimize()

                    # Record times.
                    wall_time = time.perf_counter() - wall_time
                    process_time = time.process_time() - process_time

                    # Record SCIP statistics.
                    solve_time = model.getSolvingTime()
                    n_nodes = model.getNNodes()
                    n_lps = model.getNLPs()
                    gap = model.getGap()
                    status = model.getStatus()

                    # Write results to a CSV file.
                    writer.writerow(
                        {'selector': selector, 'difficulty': instance['difficulty'], 'instance': instance['instance'],
                         'seed': seed, 'n_nodes': n_nodes, 'n_lps': n_lps, 'solve_time': solve_time, 'gap': gap,
                         'status': status, 'wall_time': wall_time, 'process_time': process_time})
                    csvfile.flush()
                    model.freeProb()
