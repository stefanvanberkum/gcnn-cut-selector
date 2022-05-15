import csv
import importlib
import os
import pickle
import sys
import time

import numpy as np
import pyscipopt as scip
import tensorflow as tf
from numpy.random import default_rng
from pyscipopt import SCIP_RESULT
from pyscipopt.scip import Cutsel

from model import GCNN
from utils import get_state, init_scip


class GCNNCutsel(Cutsel):
    """Graph convolutional neural network (GCNN) cut selector.

    This class extends PySCIPOpt's Cutsel class for user-defined cut selector plugins.

    Methods
    =======
    - :meth:`cutselselect`: This method is called whenever cuts need to be ranked.

    :ivar episode: The episode number (instance/seed combination).
    :ivar instance: The filepath to the current instance.
    :ivar out_queue: The out queue where the sampling agent should send samples to.
    :ivar out_dir: The save file path for samples.
    :ivar seed: A seed value for the random number generator.
    :ivar p_expert: The probability of querying the expert on each cut selection round.
    :ivar p_max: The maximum parallelism for low-quality cuts.
    :ivar p_max_ub: The maximum parallelism for high-quality cuts.
    :ivar skip_factor: The factor that determines the high-quality threshold relative to the highest-quality cut.
    """

    def __init__(self, model: GCNN, parameters: str, p_expert=0.05, p_max=0.1, p_max_ub=0.5, skip_factor=0.9):
        model.restore_state(parameters)

        # Compile the model call as TensorFlow function for performance.
        self.get_improvements = tf.function(model.call, input_signature=model.input_signature)

        self.p_expert = p_expert
        self.p_max = p_max
        self.p_max_ub = p_max_ub
        self.skip_factor = skip_factor

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """Samples expert (state, action) pairs based on bound improvement.

        Whenever the expert is not queried, the method falls back to hybrid cut selection rule.

        :param cuts: A list of cut candidates.
        :param forcedcuts: A list of forced cuts, that are not subject to selection.
        :param root: True if we are at the root node.
        :param maxnselectedcuts: The maximum number of selected cuts.
        :return: A dictionary of the form {'cuts': np.array, 'nselectedcuts': int, 'result': SCIP_RESULT},
            where 'cuts' represent the resorted array of cuts in descending order of cut quality, 'nselectedcuts'
            represents the number of cuts that should be selected from cuts (the first 'nselectedcuts'), and 'result'
            signals to SCIP that everything worked out.
        """

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

        state = cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, cut_edge_feats, \
                n_cons, n_vars, n_cuts

        # Get the predicted bound improvements.
        quality = self.get_improvements(state, tf.convert_to_tensor(False)).numpy()

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
    evaluate_model('setcov', seeds)
    evaluate_model('combauc', seeds)
    evaluate_model('capfac', seeds)
    evaluate_model('indset', seeds)
    print("Done!")


def evaluate_model(problem: str, seeds: np.array):
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

    instances += [{'type': 'easy', 'path': f"data/instances/{problem_folders[0]}/instance_{i + 1}.lp"} for i in
                  range(20)]
    instances += [{'type': 'medium', 'path': f"data/instances/{problem_folders[1]}/instance_{i + 1}.lp"} for i in
                  range(20)]
    instances += [{'type': 'hard', 'path': f"data/instances/{problem_folders[2]}/instance_{i + 1}.lp"} for i in
                  range(20)]

    cut_selectors = []

    # SCIP's default hybrid cut selector.
    for seed in seeds:
        cut_selectors.append({'type': 'hybrid', 'seed': seed})

    # GCNN cut selector.
    for seed in seeds:
        cut_selectors.append({'type': 'gcnn', 'seed': seed,
                              'parameters': f'trained_models/{args.problem}/{model}/{seed}/best_params.pkl'})

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}
    for policy in branching_policies:
        if policy['type'] == 'gcnn':
            if policy['name'] not in loaded_models:
                sys.path.insert(0, os.path.abspath(f"models/{policy['name']}"))
                import model

                importlib.reload(model)
                loaded_models[policy['name']] = model.GCNPolicy()
                del sys.path[0]
            policy['model'] = loaded_models[policy['name']]

    # load ml-competitor models
    for policy in branching_policies:
        if policy['type'] == 'ml-competitor':
            try:
                with open(f"{policy['model']}/normalization.pkl", 'rb') as f:
                    policy['feat_shift'], policy['feat_scale'] = pickle.load(f)
            except:
                policy['feat_shift'], policy['feat_scale'] = 0, 1

            with open(f"{policy['model']}/feat_specs.pkl", 'rb') as f:
                policy['feat_specs'] = pickle.load(f)

            if policy['name'].startswith('svmrank'):
                policy['model'] = svmrank.Model().read(f"{policy['model']}/model.txt")
            else:
                with open(f"{policy['model']}/model.pkl", 'rb') as f:
                    policy['model'] = pickle.load(f)

    print("running SCIP...")

    cut_selectors = ['hybrid', 'gcnn']

    fieldnames = ['selector', 'seed', 'type', 'instance', 'n_nodes', 'n_lps', 'solve_time', 'gap', 'status',
                  'wall_time', 'process_time']
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            for selector in cut_selectors:  # Fix
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    tf.random.set_seed(rng.integers(np.iinfo(int).max))
                    scip_seed = rng.integers(2147483648)

                    # Solve using SCIP's default hybrid cut selector.
                    model = scip.Model()
                    init_scip(model, scip_seed, cpu_time=True)
                    model.readProblem(f"{instance['path']}")

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
                    writer.writerow({'selector': f"{policy['type']}:{policy['name']}", 'seed': policy['seed'],
                                     'type': instance['type'], 'instance': instance['path'], 'nnodes': nnodes,
                                     'nlps': nlps, 'stime': stime, 'gap': gap, 'status': status, 'ndomchgs': ndomchgs,
                                     'ncutoffs': ncutoffs, 'walltime': walltime, 'proctime': proctime, })

                    csvfile.flush()
                    m.freeProb()

                    print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} ("
                          f"{nnodes + 2 * (ndomchgs + ncutoffs)}) nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall "
                          f"{proctime:.2f} proc) s. {status}")
