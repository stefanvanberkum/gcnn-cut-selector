import gzip
import importlib
import os
import pathlib
import pickle
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.optimizers import Adam

from model import BaseModel, GCNN
from utils import load_batch_tf, write_log


def train_model(problem: str, seed: int, max_epochs=1000, epoch_size=312, batch_size=32, pretrain_batch_size=128,
                valid_batch_size=128, lr=0.001, patience=10, early_stopping=20):
    fractions = [0.25, 0.5, 0.75, 1]

    problem_folders = {'setcov': 'setcov/500r', 'combauc': 'combauc/100i_500b', 'capfac': 'capfac/100c',
                       'indset': 'indset/500n', }
    problem_folder = problem_folders[problem]

    running_dir = f"trained_models/{problem}/{seed}"

    os.makedirs(running_dir)

    # Create log file.
    logfile = os.path.join(running_dir, 'log.txt')

    write_log(f"max_epochs: {max_epochs}", logfile)
    write_log(f"epoch_size: {epoch_size}", logfile)
    write_log(f"batch_size: {batch_size}", logfile)
    write_log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    write_log(f"valid_batch_size : {valid_batch_size}", logfile)
    write_log(f"lr: {lr}", logfile)
    write_log(f"patience : {patience}", logfile)
    write_log(f"early_stopping : {early_stopping}", logfile)
    write_log(f"fractions: {fractions}", logfile)
    write_log(f"problem: {problem}", logfile)
    write_log(f"seed {seed}", logfile)

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
    for epoch in range(max_epochs + 1):
        write_log(f"EPOCH {epoch}...", logfile)

        if epoch == 0:
            # Run pretraining in the first epoch.
            n = pretrain(model=model, dataloader=pretrain_data)
            write_log(f"PRETRAINED {n} LAYERS", logfile)
            # model compilation
            model.call = tfe.defun(model.call, input_signature=model.input_signature)
        else:
            # bugfix: tensorflow's shuffle() seems broken...
            epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)
            train_data = tf.data.Dataset.from_tensor_slices(epoch_train_files)
            train_data = train_data.batch(batch_size)
            train_data = train_data.map(load_batch_tf)
            train_data = train_data.prefetch(1)
            train_loss, train_kacc = process(model, train_data, top_k, optimizer)
            write_log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join(
                [f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # TEST
        valid_loss, valid_kacc = process(model, valid_data, top_k, None)
        write_log(
            f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]),
            logfile)

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(os.path.join(running_dir, 'best_params.pkl'))
            write_log(f"  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % early_stopping == 0:
                write_log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
            if plateau_count % patience == 0:
                lr *= 0.2
                write_log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)

    model.restore_state(os.path.join(running_dir, 'best_params.pkl'))
    valid_loss, valid_kacc = process(model, valid_data, top_k, None)
    write_log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join(
        [f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)


def pretrain(model: BaseModel, dataloader: tf.data.Dataset):
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


def process(model, dataloader, top_k, optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples = 0
    for batch in dataloader:
        (cons_feats, cons_edge_inds, cons_edge_feats, var_feats, cut_feats, cut_edge_inds, cut_edge_feats, n_cons,
         n_vars, n_cuts, improvements) = batch
        batched_states = (
            c, ei, ev, v, tf.reduce_sum(n_cs, keepdims=True), tf.reduce_sum(n_vs, keepdims=True))  # prevent padding
        batch_size = len(n_cs.numpy())

        if optimizer:
            with tf.GradientTape() as tape:
                logits = model(batched_states, tf.convert_to_tensor(True))  # training mode
                logits = tf.expand_dims(tf.gather(tf.squeeze(logits, 0), cands), 0)  # filter candidate variables
                logits = model.pad_output(logits, n_cands.numpy())  # apply padding now
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
        mean_kacc += kacc * batch_size
        n_samples += batch_size

    mean_loss /= n_samples
    mean_kacc /= n_samples

    return mean_loss, mean_kacc
