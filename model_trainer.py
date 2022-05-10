import argparse
import gzip
import importlib
import os
import pathlib
import pickle
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import utilities
from utilities import log

from model import BaseModel
from utils import load_batch_tf


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', help='MILP instance type to process.',
                        choices=['setcover', 'cauctions', 'facilities', 'indset'], )
    parser.add_argument('-m', '--model', help='GCNN model to be trained.', type=str, default='baseline', )
    parser.add_argument('-s', '--seed', help='Random generator seed.', type=utilities.valid_seed, default=0, )
    parser.add_argument('-g', '--gpu', help='CUDA GPU id (-1 for CPU).', type=int, default=0, )
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    max_epochs = 1000
    epoch_size = 312
    batch_size = 32
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr = 0.001
    patience = 10
    early_stopping = 20
    top_k = [1, 3, 5, 10]
    train_ncands_limit = np.inf
    valid_ncands_limit = np.inf

    problem_folders = {'setcover': 'setcover/500r_1000c_0.05d', 'cauctions': 'cauctions/100_500',
                       'facilities': 'facilities/100_100_5', 'indset': 'indset/500_4', }
    problem_folder = problem_folders[args.problem]

    running_dir = f"trained_models/{args.problem}/{args.model}/{args.seed}"

    os.makedirs(running_dir)

    ### LOG ###
    logfile = os.path.join(running_dir, 'log.txt')

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"epoch_size: {epoch_size}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size}", logfile)
    log(f"lr: {lr}", logfile)
    log(f"patience : {patience}", logfile)
    log(f"early_stopping : {early_stopping}", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    ### NUMPY / TENSORFLOW SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config)
    tf.executing_eagerly()

    rng = np.random.default_rng(args.seed)
    tf.set_random_seed(rng.integers(np.iinfo(int).max))

    ### SET-UP DATASET ###
    train_files = list(pathlib.Path(f'data/samples/{problem_folder}/train').glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f'data/samples/{problem_folder}/valid').glob('sample_*.pkl'))


    def take_subset(sample_files, cands_limit):
        nsamples = 0
        ncands = 0
        for filename in sample_files:
            with gzip.open(filename, 'rb') as file:
                sample = pickle.load(file)

            _, _, _, cands, _ = sample['data']
            ncands += len(cands)
            nsamples += 1

            if ncands >= cands_limit:
                log(f"  dataset size limit reached ({cands_limit} candidate variables)", logfile)
                break

        return sample_files[:nsamples]


    if train_ncands_limit < np.inf:
        train_files = take_subset(rng.permutation(train_files), train_ncands_limit)
    log(f"{len(train_files)} training samples", logfile)
    if valid_ncands_limit < np.inf:
        valid_files = take_subset(valid_files, valid_ncands_limit)
    log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]

    valid_data = tf.data.Dataset.from_tensor_slices(valid_files)
    valid_data = valid_data.batch(valid_batch_size)
    valid_data = valid_data.map(load_batch_tf)
    valid_data = valid_data.prefetch(1)

    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    pretrain_data = tf.data.Dataset.from_tensor_slices(pretrain_files)
    pretrain_data = pretrain_data.batch(pretrain_batch_size)
    pretrain_data = pretrain_data.map(load_batch_tf)
    pretrain_data = pretrain_data.prefetch(1)

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model

    importlib.reload(model)
    model = model.GCNPolicy()
    del sys.path[0]

    ### TRAINING LOOP ###
    optimizer = tf.train.AdamOptimizer(learning_rate=lambda: lr)  # dynamic LR trick
    best_loss = np.inf
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        # TRAIN
        if epoch == 0:
            n = pretrain(model=model, dataloader=pretrain_data)
            log(f"PRETRAINED {n} LAYERS", logfile)
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
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join(
                [f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # TEST
        valid_loss, valid_kacc = process(model, valid_data, top_k, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]),
            logfile)

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(os.path.join(running_dir, 'best_params.pkl'))
            log(f"  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % early_stopping == 0:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
            if plateau_count % patience == 0:
                lr *= 0.2
                log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)

    model.restore_state(os.path.join(running_dir, 'best_params.pkl'))
    valid_loss, valid_kacc = process(model, valid_data, top_k, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join(
        [f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
