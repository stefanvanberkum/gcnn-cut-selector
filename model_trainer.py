import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from model import GCNN
from utils import load_batch_tf, write_log


def train_model(problem: str, seed: int, max_epochs=1000, epoch_size=312, batch_size=32, pretrain_batch_size=128,
                valid_batch_size=128, lr=0.001, patience=10, early_stopping=20):
    """Trains the model.

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

    fractions = [0.25, 0.5, 0.75, 1]

    problem_folders = {'setcov': 'setcov/500r', 'combauc': 'combauc/100i_500b', 'capfac': 'capfac/100c',
                       'indset': 'indset/500n', }
    problem_folder = problem_folders[problem]

    running_dir = f"trained_models/{problem}/{seed}"

    os.makedirs(running_dir)

    # Create log file.
    logfile = os.path.join(running_dir, 'log.txt')

    write_log(f"problem: {problem}", logfile)
    write_log(f"seed {seed}", logfile)
    write_log(f"max_epochs: {max_epochs}", logfile)
    write_log(f"epoch_size: {epoch_size}", logfile)
    write_log(f"batch_size: {batch_size}", logfile)
    write_log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    write_log(f"valid_batch_size : {valid_batch_size}", logfile)
    write_log(f"lr: {lr}", logfile)
    write_log(f"patience : {patience}", logfile)
    write_log(f"early_stopping : {early_stopping}", logfile)
    write_log(f"fractions: {fractions}", logfile)

    rng = np.random.default_rng(seed)
    tf.random.set_seed(rng.integers(np.iinfo(int).max))

    # Set up dataset.
    train_files = list(pathlib.Path(f'data/samples/{problem_folder}/train').glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f'data/samples/{problem_folder}/valid').glob('sample_*.pkl'))

    write_log(f"{len(train_files)} training samples", logfile)
    write_log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]

    valid_data = tf.data.Dataset.from_tensor_slices(valid_files)
    valid_data = valid_data.batch(valid_batch_size)
    valid_data = valid_data.map(load_batch_tf)
    valid_data = valid_data.prefetch(1)

    pretrain_files = [file for i, file in enumerate(train_files) if i % 10 == 0]
    pretrain_data = tf.data.Dataset.from_tensor_slices(pretrain_files)
    pretrain_data = pretrain_data.batch(pretrain_batch_size)
    pretrain_data = pretrain_data.map(load_batch_tf)
    pretrain_data = pretrain_data.prefetch(1)

    # Load model.
    model = GCNN()

    # Training loop.
    optimizer = Adam(learning_rate=lambda: lr)  # Dynamic learning rate.
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

            # Create dataset.
            train_data = tf.data.Dataset.from_tensor_slices(epoch_train_files)
            train_data = train_data.batch(batch_size)
            train_data = train_data.map(load_batch_tf)
            train_data = train_data.prefetch(1)

            # Train the model.
            train_loss, train_acc = process(model, train_data, fractions, optimizer)
            write_log(f"TRAIN LOSS: {train_loss:.3f} " + "".join(
                [f" {100 * frac:.0f}%: {100 * acc:.3f}" for frac, acc in zip(fractions, train_acc)]), logfile)

        # Test the model on the validation set.
        valid_loss, valid_acc = process(model, valid_data, fractions, None)
        write_log(f"VALID LOSS: {valid_loss:.3f} " + "".join(
            [f" {100 * frac:.0f}%: {acc:.3f}" for frac, acc in zip(fractions, valid_acc)]), logfile)

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
    valid_loss, valid_acc = process(model, valid_data, fractions, None)
    write_log(f"BEST VALID LOSS: {valid_loss:.3f} " + "".join(
        [f" {100 * frac:.0f}%: {acc:.3f}" for frac, acc in zip(fractions, valid_acc)]), logfile)


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

    model.pre_train_init()
    i = 0
    while True:
        # Prtrain the first prenorm layer that is still open for updates.
        for batch in dataloader:
            (cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, cut_edge_feats, n_cons,
             n_vars, n_cuts, _) = batch
            batched_states = cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, \
                             cut_edge_feats, n_cons, n_vars, n_cuts

            if not model.pretrain(batched_states, tf.convert_to_tensor(True)):
                # No layer receives any updates anymore.
                break

        # Find the layer we just pretrained and turn off updating for this layer.
        res = model.pre_train_next()
        if res is None:
            # We did not train anything, implying that no layers are left and that we are done.
            break
        i += 1

    return i


def process(model: GCNN, dataloader: tf.data.Dataset, fractions: list[int], optimizer=None):
    """Runs the input data through the model, training it if an optimizer is provided.

    :param model: The model.
    :param dataloader: The input dataset to process.
    :param fractions: A list of fractions to compute the accuracy over (top x% correct).
    :param optimizer: An optional optimizer used for training the model.
    :return: The mean loss and accuracy over the input data.
    """

    mean_loss = 0
    mean_acc = np.zeros(len(fractions))

    n_samples = 0
    for batch in dataloader:
        (cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, cut_edge_feats, n_cons,
         n_vars, n_cuts, improvements) = batch
        batched_states = cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, \
                         cut_edge_feats, n_cons, n_vars, n_cuts
        batch_size = len(n_cs.numpy())

        if optimizer:
            with tf.GradientTape() as tape:
                predictions = model(batched_states, tf.convert_to_tensor(True))
                loss = tf.losses.sparse_softmax_cross_entropy(labels=best_cands, logits=logits)
            grads = tape.gradient(target=loss, sources=model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
        else:
            logits = model(batched_states, tf.convert_to_tensor(False))  # eval mode
            logits = tf.expand_dims(tf.gather(tf.squeeze(logits, 0), cands), 0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands.numpy())  # apply padding now
            loss = tf.losses.sparse_softmax_cross_entropy(labels=best_cands, logits=logits)

        true_scores = model.pad_output(tf.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = tf.reduce_max(true_scores, axis=-1, keepdims=True)
        true_scores = true_scores.numpy()
        true_bestscore = true_bestscore.numpy()

        # Measure how often it ranks the highest-quality cuts correctly (10%, 25%, 50%?)

        kacc = []
        for k in top_k:
            pred_top_k = tf.nn.top_k(logits, k=k)[1].numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
        kacc = np.asarray(kacc)

        mean_loss += loss.numpy() * batch_size
        mean_acc += kacc * batch_size
        n_samples += batch_size

    mean_loss /= n_samples
    mean_acc /= n_samples

    return mean_loss, mean_acc
