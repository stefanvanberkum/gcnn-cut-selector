"""This module provides methods for testing a trained GCNN model.

Summary
=======
This module provides methods for testing a trained GCNN model. The methods in this module are based the code by [1]_.

Functions
=========
- :func:`test_models`: Tests the models in accordance with our testing scheme.
- :func:`test_model`: Tests a trained model on testing data and writes the results to a CSV file.
- :func:`process`: Runs the input data through a trained model.

References
==========
.. [1] Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). Exact combinatorial optimization with
    graph convolutional neural networks. *Neural Information Processing Systems (NeurIPS 2019)*, 15580–15592.
    https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
"""

import csv
import gzip
import os
import pathlib
import pickle
from argparse import ArgumentParser
from datetime import timedelta
from math import ceil
from time import perf_counter

import numpy as np
import tensorflow as tf
from numpy.random import default_rng

from model import GCNN
from utils import load_batch_tf, load_seeds


def test_models():
    """Tests the models in accordance with our testing scheme."""

    seeds = load_seeds(name='train_seeds')[:2]

    print("Testing models...")
    problems = ['setcov', 'combauc', 'capfac', 'indset']
    for problem in problems:
        for i in range(5):
            print(f"Testing a model for {problem} problems, iteration {i}...")
            test_model(problem, seeds[i])


def test_model(problem: str, seed: np.array, test_batch_size=64):
    """Tests a trained model on testing data and writes the results to a CSV file.

    The accuracy on given fractions of the cut candidate ranking is written to a CSV file. That is, how often the
    model ranked the top x% of cut candidates correctly. Besides this, the hybrid baseline cut selector is also
    tested for comparison.

    :param problem: The problem type to be considered, one of: {'setcov', 'combauc', 'capfac', or 'indset'}.
    :param seed: A seed that was used for training a models.
    :param test_batch_size: The number of samples in each testing batch.
    """

    # Start timer.
    wall_start = perf_counter()

    fractions = np.array([0.25, 0.5, 0.75, 1])

    problem_folders = {'setcov': 'setcov/500r', 'combauc': 'combauc/100i_500b', 'capfac': 'capfac/100c',
                       'indset': 'indset/500n'}
    problem_folder = problem_folders[problem]

    os.makedirs(f"results/test/{problem}", exist_ok=True)
    result_file = f"results/test/{problem}/{seed}.csv"

    # Retrieve testing samples.
    test_files = list(pathlib.Path(f"data/samples/{problem_folder}/test").glob('sample_*.pkl'))
    test_files = [str(x) for x in test_files]

    # Compile the model call as TensorFlow function for performance.
    model = GCNN()
    model.call = tf.function(model.call, input_signature=model.input_signature)

    rng = np.random.default_rng(seed)
    tf.random.set_seed(rng.integers(np.iinfo(int).max))

    # Load the trained model.
    model.restore_state(f"trained_models/{problem}/{seed}/best_params.pkl")

    # Prepare testing dataset.
    test_data = tf.data.Dataset.from_tensor_slices(test_files)
    test_data = test_data.batch(test_batch_size)
    test_data = test_data.map(load_batch_tf)
    test_data = test_data.prefetch(2)

    # Test the model.
    test_acc = process(model, test_data, fractions)

    # Test the hybrid baseline cut selector.
    baseline_acc = np.zeros(len(fractions))

    # Load samples.
    for filename in test_files:
        with gzip.open(filename, 'rb') as file:
            sample = pickle.load(file)

            # Load (state, action) pair.
            sample_state, sample_improvements = sample['data']

            # Retrieve cut features.
            sample_cons, sample_cons_edge, sample_var, sample_cut, sample_cut_edge = sample_state
            cut_feats = sample_cut['values']
            cut_feat_names = sample_cut['features']
            int_support = cut_feats[:, cut_feat_names.index('int_support')]
            efficacy = cut_feats[:, cut_feat_names.index('efficacy')]
            cutoff = cut_feats[:, cut_feat_names.index('cutoff')]
            parallelism = cut_feats[:, cut_feat_names.index('parallelism')]

            # Compute cut quality scores.
            if np.any(cutoff):
                pred = efficacy + 0.1 * int_support + 0.1 * parallelism + 0.5 * cutoff
            else:
                pred = 1.5 * efficacy + 0.1 * int_support + 0.1 * parallelism

            # Sort the cut indices based on the predicted and true bound improvements.
            true = sample_improvements
            pred_ranking = np.array(sorted(range(len(pred)), key=lambda x: pred[x], reverse=True))
            true_ranking = np.array(sorted(range(len(true)), key=lambda x: true[x], reverse=True))

            # Find the first index that deviates.
            differences = (pred_ranking != true_ranking)
            if np.any(differences):
                deviation = np.argmax(pred_ranking != true_ranking)
            else:
                # No deviations.
                deviation = len(pred)

            # Compute the fraction of cuts that were ranked correctly and record it in the accuracy matrix.
            frac = deviation / len(pred)
            baseline_acc += (frac >= fractions)
    baseline_acc /= len(test_files)

    fieldnames = ['type', 'seed', ] + [f'{100 * frac:.0f}%' for frac in fractions]
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {'type': 'gcnn', 'seed': seed, **{f'{100 * k:.0f}%': test_acc[i] for i, k in enumerate(fractions)}})
        writer.writerow(
            {'type': 'baseline', 'seed': seed, **{f'{100 * k:.0f}%': baseline_acc[i] for i, k in enumerate(fractions)}})

    print("Done!")
    print(f"Wall time: {str(timedelta(seconds=ceil(perf_counter() - wall_start)))}")
    print("")


def process(model: GCNN, dataloader: tf.data.Dataset, fractions: np.array):
    """Runs the input data through a trained model.

    :param model: The trained model.
    :param dataloader: The input dataset to process.
    :param fractions: A list of fractions to compute the accuracy over (top x% correct).
    :return: The mean accuracy over the input data.
    """

    mean_acc = np.zeros(len(fractions))

    n_samples = 0
    for batch in dataloader:
        (cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, cut_edge_feats, n_cons,
         n_vars, n_cuts, improvements) = batch

        n_cons_total = tf.reduce_sum(n_cons)
        n_vars_total = tf.reduce_sum(n_vars)
        n_cuts_total = tf.reduce_sum(n_cuts)
        batched_states = cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, \
                         cut_edge_feats, n_cons_total, n_vars_total, n_cuts_total
        batch_size = len(n_cons.numpy())

        predictions = model(batched_states, tf.convert_to_tensor(False))

        # Measure how often the model ranks the highest-quality cuts correctly.
        predictions = tf.split(value=predictions, num_or_size_splits=n_cuts)
        improvements = tf.split(value=improvements, num_or_size_splits=n_cuts)

        acc = np.zeros(len(fractions))
        for i in range(batch_size):
            pred = predictions[i].numpy()
            true = improvements[i].numpy()

            # Sort the cut indices based on the predicted and true bound improvements.
            pred_ranking = np.array(sorted(range(len(pred)), key=lambda x: pred[x], reverse=True))
            true_ranking = np.array(sorted(range(len(true)), key=lambda x: true[x], reverse=True))

            # Find the first index that deviates.
            differences = (pred_ranking != true_ranking)
            if np.any(differences):
                deviation = np.argmax(pred_ranking != true_ranking)
            else:
                # No deviations.
                deviation = len(pred)

            # Compute the fraction of cuts that were ranked correctly and record it in the accuracy matrix.
            frac = deviation / len(pred)
            acc += (frac >= fractions)

        mean_acc += acc
        n_samples += batch_size

    mean_acc /= n_samples

    return mean_acc


if __name__ == '__main__':
    # For command line use.
    parser = ArgumentParser()
    parser.add_argument('problem', help='The problem type, one of {setcov, combauc, capfac, indset}.')
    parser.add_argument('iteration', help='The iteration to train (seed index), one of {1, ..., 5}.', type=int)
    args = parser.parse_args()

    # Retrieve training seeds and get the one that corresponds to the specified problem.
    test_seeds = load_seeds(name='train_seeds')
    test_seed = test_seeds[args.iteration - 1]

    # Train the model.
    print(f"Testing a model for {args.problem} problems, iteration {args.iteration}...")
    test_model(args.problem, test_seed)
