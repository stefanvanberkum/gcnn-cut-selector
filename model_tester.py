"""This module provides methods for testing a trained GCNN model.

Summary
=======
This module provides methods for testing a trained GCNN model. The methods in this module are based the code by [1]_.

Functions
=========
- :func:`test_models`: Test the models in accordance with our testing scheme.
- :func:`test_model`: Test a trained model on testing data and write the results to a CSV file.
- :func:`process`: Run the input data through a trained model.

References
==========
.. [1] Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). Exact combinatorial optimization with
    graph convolutional neural networks. *Neural Information Processing Systems (NeurIPS 2019)*, 15580–15592.
    https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
"""

import csv
import glob
import gzip
import os
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
    """Test the models in accordance with our testing scheme."""

    seeds = load_seeds(name='train_seeds')

    print("Testing models...")
    problems = ['setcov', 'combauc', 'capfac', 'indset']
    for problem in problems:
        for i in range(5):
            print(f"Testing a model for {problem} problems, iteration {i + 1}...")
            test_model(problem, seeds[i])


def test_model(problem: str, seed: np.array, test_batch_size=4):
    """Test a trained model on testing data and write the results to a CSV file.

    The average accuracy of the cut candidate ranking is written to a CSV file (i.e., how many cut candidates the
    model ranked correctly on average). Besides this, a baseline that randomly ranks cuts and the hybrid cut selector
    are also tested for comparison.

    :param problem: The problem type to be considered, one of: {'setcov', 'combauc', 'capfac', or 'indset'}.
    :param seed: A seed that was used for training a models.
    :param test_batch_size: The number of samples in each testing batch.
    """

    # Start timer.
    wall_start = perf_counter()

    problem_folders = {'setcov': 'setcov/500r', 'combauc': 'combauc/100i_500b', 'capfac': 'capfac/100c_100f',
                       'indset': 'indset/500n'}
    problem_folder = problem_folders[problem]

    os.makedirs(f"results/test/{problem}", exist_ok=True)
    result_file = f"results/test/{problem}/{seed}.csv"
    loss_file = f"results/test/{problem}/{seed}_loss"

    # Retrieve testing samples.
    test_files = sorted(glob.glob(f"data/samples/{problem_folder}/test/sample_*.pkl"))

    # Disable TensorFlow logs.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Compile the model call as TensorFlow function for performance.
    model = GCNN()
    model.call = tf.function(model.call, input_signature=model.input_signature)

    rng = np.random.default_rng(seed)
    tf.random.set_seed(int(rng.integers(np.iinfo(int).max)))

    # Load the trained model.
    model.restore_state(f"trained_models/{problem}/{seed}/best_params.pkl")

    # Prepare testing dataset.
    test_data = tf.data.Dataset.from_tensor_slices(test_files)
    test_data = test_data.batch(test_batch_size)
    test_data = test_data.map(load_batch_tf)
    test_data = test_data.prefetch(2)

    # Test the model.
    test_loss, test_acc = process(model, test_data)

    # Test the baseline and hybrid cut selector.
    random_acc = 0
    hybrid_acc = 0

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
            parallelism = cut_feats[:, cut_feat_names.index('parallelism')]

            # Compute cut quality scores.
            pred = efficacy + 0.1 * int_support + 0.1 * parallelism

            # Sort the cut indices based on the predicted and true bound improvements.
            true = sample_improvements
            pred_ranking = np.array(sorted(range(len(pred)), key=lambda x: pred[x], reverse=True))
            true_ranking = np.array(sorted(range(len(true)), key=lambda x: true[x], reverse=True))

            # Find the first index that deviates for the baseline.
            random_ranking = np.arange(len(true))
            rng.shuffle(random_ranking)
            differences = (random_ranking != true_ranking)
            if np.any(differences):
                deviation = np.argmax(random_ranking != true_ranking)
            else:
                # No deviations.
                deviation = len(true)

            # Compute the fraction of cuts that were ranked correctly and record it in the accuracy matrix.
            frac = deviation / len(true)
            random_acc += frac

            # Find the first index that deviates for the hybrid cut selector.
            differences = (pred_ranking != true_ranking)
            if np.any(differences):
                deviation = np.argmax(pred_ranking != true_ranking)
            else:
                # No deviations.
                deviation = len(true)

            # Compute the fraction of cuts that were ranked correctly and record it in the accuracy matrix.
            frac = deviation / len(true)
            hybrid_acc += frac
    random_acc /= len(test_files)
    hybrid_acc /= len(test_files)

    fieldnames = ['type', 'seed', 'fraction']
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'type': 'random', 'seed': seed, 'fraction': random_acc})
        writer.writerow({'type': 'hybrid', 'seed': seed, 'fraction': hybrid_acc})
        writer.writerow({'type': 'gcnn', 'seed': seed, 'fraction': test_acc})

    # Record loss.
    np.save(loss_file, np.array(test_loss))

    tf.keras.backend.clear_session()

    print("Done!")
    print(f"Wall time: {str(timedelta(seconds=ceil(perf_counter() - wall_start)))}")
    print("")


def process(model: GCNN, dataloader: tf.data.Dataset):
    """Run the input data through a trained model.

    :param model: The trained model.
    :param dataloader: The input dataset to process.
    :return: The loss and mean accuracies over the input data.
    """

    loss = 0
    mean_acc = 0

    n_samples = 0
    cut_count = 0
    for batch in dataloader:
        (cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, cut_edge_feats, n_cons,
         n_vars, n_cuts, improvements) = batch

        n_cons_total = tf.reduce_sum(n_cons)
        n_vars_total = tf.reduce_sum(n_vars)
        n_cuts_total = tf.reduce_sum(n_cuts)
        batched_states = cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, \
                         cut_edge_feats, n_cons_total, n_vars_total, n_cuts_total
        batch_size = len(n_cons.numpy())

        try:
            predictions = model(batched_states, tf.convert_to_tensor(False))
            loss += n_cuts_total.numpy() * tf.keras.metrics.mean_squared_error(improvements, predictions).numpy()

            # Measure how often the model ranks the highest-quality cuts correctly.
            predictions = tf.split(value=predictions, num_or_size_splits=n_cuts)
            improvements = tf.split(value=improvements, num_or_size_splits=n_cuts)

            acc = 0
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
                acc += frac

            mean_acc += acc
            n_samples += batch_size
            cut_count += n_cuts_total.numpy()
        except tf.errors.ResourceExhaustedError:
            # Skip batch if it's too large.
            print("WARNING: batch skipped.")
            pass

    loss /= cut_count
    mean_acc /= n_samples

    return loss, mean_acc


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
