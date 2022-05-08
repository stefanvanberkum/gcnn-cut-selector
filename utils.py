"""This module provides some useful helper methods.

Summary
=======
This module provides methods for randomly generated set covering, combinatorial auction, capacitated facility location,
and maximum independent set problem instances. The methods in this module are based on [1]_.

Classes
========
- :class:`Graph`: Data type for a general graph structure with methods for random graph generation.

Functions
=========
- :func:`generate_setcov`: Generates a random set cover problem instance.
- :func:`generate_combauc`: Generates a random combinatorial auction problem instance.
- :func:`generate_capfac`: Generates a random capacitated facility location problem instance.
- :func:`generate_indset`: Generates a random maximum independent set problem instance.

References
==========
.. [1] Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). Exact combinatorial optimization with
    graph convolutional neural networks. *Neural Information Processing Systems (NeurIPS 2019)*, 15580–15592.
    https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
"""

import datetime
import gzip
import pickle
from math import floor

import numpy as np
import pyscipopt.scip
import scipy.sparse as sp
import tensorflow as tf


def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def get_state(model: pyscipopt.scip.Model, cuts: list[pyscipopt.scip.Row]):
    """Extracts the graph representation of the problem at the current solver state.

    The nodes in this graph are the constraints, variables, and cut candidates. Constraints and cuts are connected to
    a variable if and only if this variable appears in the row (cut or constraint).

    :param model: The current model.
    :param cuts: The current list of cut candidates.
    :return: A tuple consisting of the constraint, constraint edge, variable, cut, and cut edge features. The
        constraint, variable, and cut features are dictionaries of the form {'features': list[str], 'values':
        np.ndarray}. The edge features are of the form {'features': list[str], 'indices': np.ndarray, 'values':
        np.ndarray}, where the values are provided in COO sparse matrix format.
    """

    # Compute the norm of the objective value.
    obj_norm = np.linalg.norm(list(model.getObjective().terms.values()))
    obj_norm = 1 if obj_norm <= 0 else obj_norm

    # Retrieve rows (constraints) and columns (variables).
    rows = model.getLPRowsData()
    cols = model.getLPColsData()
    n_rows = len(rows)
    n_cols = len(cols)
    n_cuts = len(cuts)

    # Row (constraint) features.
    row_feats = {}

    # Compute the norm of each constraint.
    row_norms = np.array([row.getNorm() for row in rows])
    row_norms[row_norms == 0] = 1

    # Split constraints of the form lhs <= d^T x <= rhs into two parts (lhs <= d^T x is transformed to -d^T x <= -lhs).
    lhs = np.array([row.getLhs() for row in rows])
    rhs = np.array([row.getRhs() for row in rows])
    has_lhs = [not model.isInfinity(-val) for val in lhs]
    has_rhs = [not model.isInfinity(val) for val in rhs]
    rows = np.array(rows)

    # Compute the right-hand side of each constraint, normalized by the row norm.
    row_feats['rhs'] = np.concatenate((-(lhs / row_norms)[has_lhs], (rhs / row_norms)[has_rhs])).reshape(-1, 1)

    # Compute tightness indicator.
    row_feats['is_tight'] = np.concatenate(([row.getBasisStatus() == 'lower' for row in rows[has_lhs]],
                                            [row.getBasisStatus() == 'upper' for row in rows[has_rhs]])).reshape(-1, 1)

    # Compute cosine similarity with the objective function.
    cosines = np.array([get_objCosine(rows[i], row_norms[i], obj_norm) for i in range(n_rows)])
    row_feats['obj_cosine'] = np.concatenate((-cosines[has_lhs], cosines[has_rhs])).reshape(-1, 1)

    # Compute the dual solution value, normalized by the product of the row and objective norm.
    duals = np.array([model.getRowDualSol(row) for row in rows]) / (row_norms * obj_norm)
    row_feats['dual'] = np.concatenate((-duals[has_lhs], +duals[has_rhs])).reshape(-1, 1)

    row_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                      row_feats.items()]
    row_feat_names = [n for names in row_feat_names for n in names]
    row_feat_vals = np.concatenate(list(row_feats.values()), axis=-1)
    row_feats = {'features': row_feat_names, 'values': row_feat_vals}

    # Constraint edge features.
    # For each row, record a vector [value / row_norm, row_index, column_index] and stack everything into one big
    # matrix (-1x3).
    data = np.array([[rows[i].getVals()[j] / row_norms[i], rows[i].getLPPos(), rows[i].getCols()[j].getLPPos()] for i in
                     range(n_rows) for j in range(len(rows[i].getCols()))])

    # Put into sparse CSR matrix format, transform to COO format, and collect indices.
    coef_matrix = sp.csr_matrix((data[:, 0], (data[:, 1], data[:, 2])), shape=(n_rows, n_cols))
    coef_matrix = sp.vstack((-coef_matrix[has_lhs, :], coef_matrix[has_rhs, :])).tocoo(copy=False)
    row_ind, col_ind = coef_matrix.row, coef_matrix.col
    edge_feats = {'coef': coef_matrix.data.reshape(-1, 1)}

    edge_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                       edge_feats.items()]
    edge_feat_names = [n for names in edge_feat_names for n in names]
    edge_feat_indices = np.vstack([row_ind, col_ind])
    edge_feat_vals = np.concatenate(list(edge_feats.values()), axis=-1)
    row_edge_feats = {'features': edge_feat_names, 'indices': edge_feat_indices, 'values': edge_feat_vals}

    # Column (variable) features.
    col_feats = {}

    # Retrieve column type.
    type_map = {'BINARY': 0, 'INTEGER': 1, 'IMPLINT': 2, 'CONTINUOUS': 3}
    types = np.array([type_map[col.getVar().vtype()] for col in cols])
    col_feats['type'] = np.zeros((n_cols, 4))
    col_feats['type'][np.arange(n_cols), types] = 1

    # Compute normalized column coefficient in the objective function.
    col_feats['obj_coef'] = np.array([col.getObjCoeff() for col in cols]).reshape(-1, 1) / obj_norm

    # Get variable lower and upper bounds, and whether the variable is at these bounds.
    lb = np.array([col.getLb() for col in cols])
    ub = np.array([col.getUb() for col in cols])
    has_lb = [not model.isInfinity(-val) for val in lb]
    has_ub = [not model.isInfinity(val) for val in ub]
    col_feats['has_lb'] = np.array(has_lb).astype(int).reshape(-1, 1)
    col_feats['has_ub'] = np.array(has_ub).astype(int).reshape(-1, 1)
    col_feats['at_lb'] = np.array([col.getBasisStatus() == 'lower' for col in cols]).reshape(-1, 1)
    col_feats['at_ub'] = np.array([col.getBasisStatus() == 'upper' for col in cols]).reshape(-1, 1)

    # Get variable fractionality in the current LP solution.
    col_feats['frac'] = np.array(
        [0.5 - abs(col.getVar().getLPSol() - floor(col.getVar().getLPSol()) - 0.5) for col in cols]).reshape(-1, 1)
    col_feats['frac'][types == 3] = 0  # Continuous variables have no fractionality.

    # Compute the normalized reduced cost.
    col_feats['reduced_cost'] = np.array([model.getVarRedcost(col.getVar()) for col in cols]).reshape(-1, 1) / obj_norm

    # Get the variable's value in the current LP solution.
    col_feats['lp_val'] = np.array([col.getVar().getLPSol() for col in cols]).reshape(-1, 1)

    # Get the variable's value in the current primal solution.
    col_feats['primal_val'] = np.array([col.getPrimsol() for col in cols]).reshape(-1, 1)

    # Compute the variable's average value over all primal solutions.
    col_feats['avg_primal'] = np.mean([[model.getSolVal(sol, col.getVar()) for sol in model.getSols()] for col in cols],
                                      axis=1).reshape(-1, 1)

    col_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                      col_feats.items()]
    col_feat_names = [n for names in col_feat_names for n in names]
    col_feat_vals = np.concatenate(list(col_feats.values()), axis=-1)

    col_feats = {'features': col_feat_names, 'values': col_feat_vals}

    # Cut candidate features.
    cut_feats = {}

    # Compute the norm of each cut candidate.
    cut_norms = np.array([cut.getNorm() for cut in cuts])
    cut_norms[cut_norms == 0] = 1

    # Retrieve the right-hand side of the cut (cuts of the form lhs <= d^T x are transformed to -d^T x <= -lhs).
    # If the cut is of the form lhs <= d^T x <= rhs, we take the most binding side.
    activity = np.array([model.getRowActivity(cut) for cut in cuts])
    lhs = np.array([cut.getLhs() for cut in cuts])
    rhs = np.array([cut.getRhs() for cut in cuts])
    has_lhs = (activity - lhs) < (rhs - activity)
    has_rhs = np.logical_not(has_lhs)

    # Compute the right-hand side of each cut candidate, normalized by the cut norm.
    cut_feats['rhs'] = np.concatenate((-(lhs / cut_norms)[has_lhs], (rhs / cut_norms)[has_rhs])).reshape(-1, 1)

    # Compute each cut's support.
    support = np.array([cut.getNNonz() for cut in cuts]) / model.getNVars()
    cut_feats['support'] = np.concatenate(support[has_lhs], support[has_rhs]).reshape(-1, 1)

    # Compute each cut's integral support.
    n_int = np.array([model.getRowNumIntCols(cut) for cut in cuts])
    int_support = n_int / support
    cut_feats['int_support'] = np.concatenate(int_support[has_lhs], int_support[has_rhs]).reshape(-1, 1)

    # Compute each cut's efficacy.
    efficacy = np.array([model.getCutEfficacy(cut) for cut in cuts])
    cut_feats['efficacy'] = np.concatenate(efficacy[has_lhs], efficacy[has_rhs]).reshape(-1, 1)

    # Compute each cut's directed cutoff distance.
    cutoff = np.array([model.getCutLPSolCutoffDistance(cut, model.getBestSol()) for cut in cuts])
    cut_feats['cutoff'] = np.concatenate(cutoff[has_lhs], cutoff[has_rhs]).reshape(-1, 1)

    # Compute each cut's objective parallelism.
    parallelism = np.array([model.getRowObjParallelism(cut) for cut in cuts])
    cut_feats['parallelism'] = np.concatenate(parallelism[has_lhs], parallelism[has_rhs]).reshape(-1, 1)

    cut_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                      cut_feats.items()]
    cut_feat_names = [n for names in cut_feat_names for n in names]
    cut_feat_vals = np.concatenate(list(cut_feats.values()), axis=-1)
    cut_feats = {'features': cut_feat_names, 'values': cut_feat_vals}

    # Cut edge features.
    # For each cut, record a vector [value / cut_norm, cut_index, column_index] and stack everything into one big
    # matrix (-1x3).
    data = np.array(
        [[cuts[i].getVals()[j] / cut_norms[i], i, cuts[i].getCols()[j].getLPPos()] for i in range(n_cuts) for j in
         range(len(cuts[i].getCols()))])

    # Put into sparse CSR matrix format, transform to COO format, and collect indices.
    coef_matrix = sp.csr_matrix((data[:, 0], (data[:, 1], data[:, 2])), shape=(n_cuts, n_cols))
    coef_matrix = sp.vstack((-coef_matrix[has_lhs, :], coef_matrix[has_rhs, :])).tocoo(copy=False)
    cut_ind, col_ind = coef_matrix.row, coef_matrix.col
    edge_feats = {'coef': coef_matrix.data.reshape(-1, 1)}

    edge_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                       edge_feats.items()]
    edge_feat_names = [n for names in edge_feat_names for n in names]
    edge_feat_indices = np.vstack([cut_ind, col_ind])
    edge_feat_vals = np.concatenate(list(edge_feats.values()), axis=-1)
    cut_edge_feats = {'features': edge_feat_names, 'indices': edge_feat_indices, 'values': edge_feat_vals}

    return row_feats, row_edge_feats, col_feats, cut_feats, cut_edge_feats


def get_objCosine(row: pyscipopt.scip.Row, row_norm: float, obj_norm: float):
    """Computes the cosine similarity between a row and the objective function.

    :param row: The row.
    :param row_norm: The norm of the row.
    :param obj_norm: The norm of the objective function.
    :return: The cosine similarity.
    """

    cols = row.getCols()
    vals = row.getVals()
    dot = np.sum([vals[i] * cols[i].getObjCoeff() for i in range(len(cols))])
    return dot / (row_norm * obj_norm)


def init_scip(model: pyscipopt.scip.Model, seed: int, cpu_clock=False):
    """Initializes the SCIP model parameters.

    :param model: The SCIP model to be initialized.
    :param seed: The desired seed value to be used for variable permutation and other random components of the solver.
    :param cpu_clock: True if CPU time should be used for timing, otherwise wall clock time will be used.
    """

    # Trim seeds that exceed SCIP's maximum seed value.
    seed = seed % 2147483648

    # Set up randomization.
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # Disable presolver restarts.
    model.setIntParam('presolving/maxrestarts', 0)

    # Disable output.
    model.setIntParam('display/verblevel', 0)

    # Set time settings.
    model.setRealParam('limits/time', 3600)
    if cpu_clock:
        model.setIntParam('timing/clocktype', 1)


def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2 ** 32 - 1:
        raise argparse.ArgumentTypeError("seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed


def compute_extended_variable_features(state, candidates):
    """
    Utility to extract variable features only from a bipartite state representation.
    Parameters
    ----------
    state : dict
        A bipartite state representation.
    candidates: list of ints
        List of candidate variables for which to compute features (given as indexes).
    Returns
    -------
    variable_states : np.array
        The resulting variable states.
    """
    constraint_features, edge_features, variable_features = state
    constraint_features = constraint_features['values']
    edge_indices = edge_features['indices']
    edge_features = edge_features['values']
    variable_features = variable_features['values']

    cand_states = np.zeros(
        (len(candidates), variable_features.shape[1] + 3 * (edge_features.shape[1] + constraint_features.shape[1]),))

    # re-order edges according to variable index
    edge_ordering = edge_indices[1].argsort()
    edge_indices = edge_indices[:, edge_ordering]
    edge_features = edge_features[edge_ordering]

    # gather (ordered) neighbourhood features
    nbr_feats = np.concatenate([edge_features, constraint_features[edge_indices[0]]], axis=1)

    # split neighborhood features by variable, along with the corresponding variable
    var_cuts = np.diff(edge_indices[1]).nonzero()[0] + 1
    nbr_feats = np.split(nbr_feats, var_cuts)
    nbr_vars = np.split(edge_indices[1], var_cuts)
    assert all([all(vs[0] == vs) for vs in nbr_vars])
    nbr_vars = [vs[0] for vs in nbr_vars]

    # process candidate variable neighborhoods only
    for var, nbr_id, cand_id in zip(*np.intersect1d(nbr_vars, candidates, return_indices=True)):
        cand_states[cand_id, :] = np.concatenate(
            [variable_features[var, :], nbr_feats[nbr_id].min(axis=0), nbr_feats[nbr_id].mean(axis=0),
             nbr_feats[nbr_id].max(axis=0)])

    cand_states[np.isnan(cand_states)] = 0

    return cand_states


def extract_khalil_variable_features(model, candidates, root_buffer):
    """
    Extract features following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.
    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    candidates : list of pyscipopt.scip.Variable's
        A list of variables for which to compute the variable features.
    root_buffer : dict
        A buffer to avoid re-extracting redundant root node information (None to deactivate buffering).
    Returns
    -------
    variable_features : 2D np.ndarray
        The features associated with the candidate variables.
    """
    # update state from state_buffer if any
    scip_state = model.getKhalilState(root_buffer, candidates)

    variable_feature_names = sorted(scip_state)
    variable_features = np.stack([scip_state[feature_name] for feature_name in variable_feature_names], axis=1)

    return variable_features


def preprocess_variable_features(features, interaction_augmentation, normalization):
    """
    Features preprocessing following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.
    Parameters
    ----------
    features : 2D np.ndarray
        The candidate variable features to preprocess.
    interaction_augmentation : bool
        Whether to augment features with 2-degree interactions (useful for linear models such as SVMs).
    normalization : bool
        Wether to normalize features in [0, 1] (i.e., query-based normalization).
    Returns
    -------
    variable_features : 2D np.ndarray
        The preprocessed variable features.
    """
    # 2-degree polynomial feature augmentation
    if interaction_augmentation:
        interactions = (np.expand_dims(features, axis=-1) * np.expand_dims(features, axis=-2)).reshape(
            (features.shape[0], -1))
        features = np.concatenate([features, interactions], axis=1)

    # query-based normalization in [0, 1]
    if normalization:
        features -= features.min(axis=0, keepdims=True)
        max_val = features.max(axis=0, keepdims=True)
        max_val[max_val == 0] = 1
        features /= max_val

    return features


def load_flat_samples(filename, feat_type, label_type, augment_feats, normalize_feats):
    with gzip.open(filename, 'rb') as file:
        sample = pickle.load(file)

    state, khalil_state, best_cand, cands, cand_scores = sample['data']

    cands = np.array(cands)
    cand_scores = np.array(cand_scores)

    cand_states = []
    if feat_type in ('all', 'gcnn_agg'):
        cand_states.append(compute_extended_variable_features(state, cands))
    if feat_type in ('all', 'khalil'):
        cand_states.append(khalil_state)
    cand_states = np.concatenate(cand_states, axis=1)

    best_cand_idx = np.where(cands == best_cand)[0][0]

    # feature preprocessing
    cand_states = preprocess_variable_features(cand_states, interaction_augmentation=augment_feats,
                                               normalization=normalize_feats)

    if label_type == 'scores':
        cand_labels = cand_scores

    elif label_type == 'ranks':
        cand_labels = np.empty(len(cand_scores), dtype=int)
        cand_labels[cand_scores.argsort()] = np.arange(len(cand_scores))

    elif label_type == 'bipartite_ranks':
        # scores quantile discretization as in
        # Khalil et al. (2016) Learning to Branch in Mixed Integer Programming
        cand_labels = np.empty(len(cand_scores), dtype=int)
        cand_labels[cand_scores >= 0.8 * cand_scores.max()] = 1
        cand_labels[cand_scores < 0.8 * cand_scores.max()] = 0

    else:
        raise ValueError(f"Invalid label type: '{label_type}'")

    return cand_states, cand_labels, best_cand_idx


def load_batch_gcnn(sample_files):
    """
    Loads and concatenates a bunch of samples into one mini-batch.
    """
    c_features = []
    e_indices = []
    e_features = []
    v_features = []
    candss = []
    cand_choices = []
    cand_scoress = []

    # load samples
    for filename in sample_files:
        with gzip.open(filename, 'rb') as f:
            sample = pickle.load(f)

        sample_state, _, sample_action, sample_cands, cand_scores = sample['data']

        sample_cands = np.array(sample_cands)
        cand_choice = np.where(sample_cands == sample_action)[0][0]  # action index relative to candidates

        c, e, v = sample_state
        c_features.append(c['values'])
        e_indices.append(e['indices'])
        e_features.append(e['values'])
        v_features.append(v['values'])
        candss.append(sample_cands)
        cand_choices.append(cand_choice)
        cand_scoress.append(cand_scores)

    n_cs_per_sample = [c.shape[0] for c in c_features]
    n_vs_per_sample = [v.shape[0] for v in v_features]
    n_cands_per_sample = [cds.shape[0] for cds in candss]

    # concatenate samples in one big graph
    c_features = np.concatenate(c_features, axis=0)
    v_features = np.concatenate(v_features, axis=0)
    e_features = np.concatenate(e_features, axis=0)
    # edge indices have to be adjusted accordingly
    cv_shift = np.cumsum([[0] + n_cs_per_sample[:-1], [0] + n_vs_per_sample[:-1]], axis=1)
    e_indices = np.concatenate([e_ind + cv_shift[:, j:(j + 1)] for j, e_ind in enumerate(e_indices)], axis=1)
    # candidate indices as well
    candss = np.concatenate([cands + shift for cands, shift in zip(candss, cv_shift[1])])
    cand_choices = np.array(cand_choices)
    cand_scoress = np.concatenate(cand_scoress, axis=0)

    # convert to tensors
    c_features = tf.convert_to_tensor(c_features, dtype=tf.float32)
    e_indices = tf.convert_to_tensor(e_indices, dtype=tf.int32)
    e_features = tf.convert_to_tensor(e_features, dtype=tf.float32)
    v_features = tf.convert_to_tensor(v_features, dtype=tf.float32)
    n_cs_per_sample = tf.convert_to_tensor(n_cs_per_sample, dtype=tf.int32)
    n_vs_per_sample = tf.convert_to_tensor(n_vs_per_sample, dtype=tf.int32)
    candss = tf.convert_to_tensor(candss, dtype=tf.int32)
    cand_choices = tf.convert_to_tensor(cand_choices, dtype=tf.int32)
    cand_scoress = tf.convert_to_tensor(cand_scoress, dtype=tf.float32)
    n_cands_per_sample = tf.convert_to_tensor(n_cands_per_sample, dtype=tf.int32)

    return c_features, e_indices, e_features, v_features, n_cs_per_sample, n_vs_per_sample, n_cands_per_sample, \
           candss, cand_choices,