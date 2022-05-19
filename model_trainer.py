"""This module provides methods for training the GCNN model.

Summary
=======
This module provides methods for training the GCNN model. The methods in this module are based the code by [1]_.

Functions
=========
- :func:`train_models`: Trains the models in accordance with our training scheme.
- :func:`train_model`: Trains a model.
- :func:`pretrain`: Pretrains a model.
- :func:`process`: Runs the input data through a model, training it if an optimizer is provided.

References
==========
.. [1] Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). Exact combinatorial optimization with
    graph convolutional neural networks. *Neural Information Processing Systems (NeurIPS 2019)*, 15580–15592.
    https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
"""
import os
import pathlib
from time import perf_counter

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from model import GCNN
from utils import load_batch_tf, load_seeds, write_log


def train_models():
    """Trains the models in accordance with our training scheme."""

    seeds = load_seeds()

    print("Training models...")
    for i in range(5):
        print(f"Iteration: {i}")
        train_model('setcov', seeds[i])
        train_model('combauc', seeds[i])
        train_model('capfac', seeds[i])
        train_model('indset', seeds[i])
    print("Done!")


def train_model(problem: str, seed: int, max_epochs=1000, epoch_size=312, batch_size=32, pretrain_batch_size=128,
                valid_batch_size=128, lr=0.001, patience=10, early_stopping=20):
    """Trains a model.

    On the first epoch, the model is pretrained. Afterwards, regular training commences. The learning rate is
    dynamically adapted using the *patience* parameter. Whenever the number of consecutive epochs without improvement
    on the validation set exceeds our *patience*, the learning rate is divided by five. An early stopping criterium
    is triggered when the number of consecutive epochs without improvement on the validation set exceeds the
    *early_stopping* parameter.

    :param problem: The problem type to be considered, one of: {'setcov', 'combauc', 'capfac', or 'indset'}.
    :param seed: A seed value for the random number generator.
    :param max_epochs: The maximum number of epochs.
    :param epoch_size: The number of batches in each epoch.
    :param batch_size: The number of samples in each training batch.
    :param pretrain_batch_size: The number of samples in each pretraining batch.
    :param valid_batch_size: The number of samples in each validation batch.
    :param lr: The initial learning rate.
    :param patience: The number of epochs without improvement required before the learning rate is adapted.
    :param early_stopping: The number of epochs without improvement required before early stopping is triggered.
    """

    fractions = np.array([0.25, 0.5, 0.75, 1])

    problem_folders = {'setcov': 'setcov/500r', 'combauc': 'combauc/100i_500b', 'capfac': 'capfac/100c',
                       'indset': 'indset/500n'}
    problem_folder = problem_folders[problem]

    running_dir = f"trained_models/{problem}/{seed}"
    os.makedirs(running_dir)

    # Create log file.
    logfile = os.path.join(running_dir, 'log.txt')

    write_log(f"problem: {problem}", logfile)
    write_log(f"seed: {seed}", logfile)
    write_log(f"max_epochs: {max_epochs}", logfile)
    write_log(f"epoch_size: {epoch_size}", logfile)
    write_log(f"batch_size: {batch_size}", logfile)
    write_log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    write_log(f"valid_batch_size: {valid_batch_size}", logfile)
    write_log(f"lr: {lr}", logfile)
    write_log(f"patience: {patience}", logfile)
    write_log(f"early_stopping: {early_stopping}", logfile)
    write_log(f"fractions: {fractions}", logfile)

    rng = np.random.default_rng(seed)
    tf.random.set_seed(rng.integers(np.iinfo(int).max))

    # Retrieve training and validation samples.
    train_files = list(pathlib.Path(f'data/samples/{problem_folder}/train').glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f'data/samples/{problem_folder}/valid').glob('sample_*.pkl'))

    write_log(f"{len(train_files)} training samples", logfile)
    write_log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]

    # Prepare validation dataset.
    valid_data = tf.data.Dataset.from_tensor_slices(valid_files)
    valid_data = valid_data.batch(valid_batch_size)
    valid_data = valid_data.map(load_batch_tf)
    valid_data = valid_data.prefetch(1)

    # Prepare pretraining dataset.
    pretrain_files = [file for i, file in enumerate(train_files) if i % 10 == 0]
    pretrain_data = tf.data.Dataset.from_tensor_slices(pretrain_files)
    pretrain_data = pretrain_data.batch(pretrain_batch_size)
    pretrain_data = pretrain_data.map(load_batch_tf)
    pretrain_data = pretrain_data.prefetch(1)

    # Initialize the model.
    model = GCNN()

    # Start timer.
    start_time = perf_counter()

    # Training loop.
    optimizer = Adam(learning_rate=lambda: lr)  # Dynamic learning rate.
    loss_fn = MeanSquaredError()
    best_loss = np.Inf
    plateau_count = 0
    for epoch in range(max_epochs + 1):
        write_log(f"EPOCH {epoch}...", logfile)

        if epoch == 0:
            # Run pretraining in the first epoch.
            n = pretrain(model=model, dataloader=pretrain_data)
            write_log(f"PRETRAINED {n} LAYERS", logfile)

            # Compile the model call as TensorFlow function for performance.
            model.call = tf.function(model.call, input_signature=model.input_signature)
        else:
            # Sample training files with replacement.
            epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)

            # Prepare training dataset.
            train_data = tf.data.Dataset.from_tensor_slices(epoch_train_files)
            train_data = train_data.batch(batch_size)
            train_data = train_data.map(load_batch_tf)
            train_data = train_data.prefetch(1)

            # Train the model.
            train_loss, train_acc = process(model, train_data, fractions, loss_fn, optimizer)
            write_log(f"TRAIN LOSS: {train_loss:.3e} " + "".join(
                [f" {100 * frac:.0f}%: {100 * acc:.3f}" for frac, acc in zip(fractions, train_acc)]), logfile)

        # Test the model on the validation set.
        valid_loss, valid_acc = process(model, valid_data, fractions, loss_fn)
        write_log(f"VALID LOSS: {valid_loss:.3e} " + "".join(
            [f" {100 * frac:.0f}%: {100 * acc:.3f}" for frac, acc in zip(fractions, valid_acc)]), logfile)

        if valid_loss < best_loss:
            # Improvement in this epoch.
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(os.path.join(running_dir, 'best_params.pkl'))
            write_log(f"  best model so far", logfile)
        else:
            # No improvement in this epoch, check whether we need to stop early or decrease the learning rate.
            plateau_count += 1
            if plateau_count % early_stopping == 0:
                write_log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
            if plateau_count % patience == 0:
                lr *= 0.2
                write_log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)

    model.restore_state(os.path.join(running_dir, 'best_params.pkl'))
    valid_loss, valid_acc = process(model, valid_data, fractions, loss_fn)
    write_log(f"BEST VALID LOSS: {valid_loss:.3e} " + "".join(
        [f" {100 * frac:.0f}%: {100 * acc:.3f}" for frac, acc in zip(fractions, valid_acc)]), logfile)
    write_log(f"Training time: {start_time - perf_counter()} seconds", logfile)


def pretrain(model: GCNN, dataloader: tf.data.Dataset):
    """Pretrains a model.

    This function pretrains the model layer-by-layer. So on each iteration, all batches are used to pretrain the
    first available prenorm layer that is still open for updates. Then, this layer is found by calling
    :func:`model.pretrain_next()`. This also turns off updates for this layer, effectively moving the pretraining
    loop to the next layer.

    :param model: The model to pretrain.
    :param dataloader: The dataset to use for pretraining.
    :return: The number of prenorm layers that have been processed.
    """

    model.pretrain_init()
    i = 0
    while True:
        # Prtrain the first prenorm layer that is still open for updates.
        for batch in dataloader:
            (cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, cut_edge_feats, n_cons,
             n_vars, n_cuts, _) = batch

            n_cons_total = tf.reduce_sum(n_cons)
            n_vars_total = tf.reduce_sum(n_vars)
            n_cuts_total = tf.reduce_sum(n_cuts)
            batched_states = cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, \
                             cut_edge_feats, n_cons_total, n_vars_total, n_cuts_total

            if not model.pretrain(batched_states, tf.convert_to_tensor(True)):
                # No layer receives any updates anymore.
                break

        # Find the layer we just pretrained and turn off updating for this layer.
        res = model.pretrain_next()
        if res is None:
            # We did not train anything, implying that no layers are left and that we are done.
            break
        i += 1

    return i


def process(model: GCNN, dataloader: tf.data.Dataset, fractions: np.array, loss_fn, optimizer=None):
    """Runs the input data through a model, training it if an optimizer is provided.

    :param model: The model.
    :param dataloader: The input dataset to process.
    :param fractions: A list of fractions to compute the accuracy over (top x% correct).
    :param loss_fn: The loss function to be used for the model.
    :param optimizer: An optional optimizer used for training the model.
    :return: The mean loss and accuracy over the input data.
    """

    mean_loss = 0
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

        if optimizer:
            # Train the model.
            with tf.GradientTape() as tape:
                predictions = model(batched_states, tf.convert_to_tensor(True))
                loss = loss_fn(improvements, predictions)
            grads = tape.gradient(target=loss, sources=model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        else:
            # Evaluate the model.
            predictions = model(batched_states, tf.convert_to_tensor(False))
            loss = loss_fn(improvements, predictions)

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

        mean_loss += loss.numpy() * batch_size
        mean_acc += acc
        n_samples += batch_size

    mean_loss /= n_samples
    mean_acc /= n_samples

    return mean_loss, mean_acc
