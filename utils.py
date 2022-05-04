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

import numpy as np
import pyscipopt.scip
import scipy.sparse as sp


def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def scip_init(model, seed):
    """Initializes the SCIP model parameters.

    :param model: The SCIP model to be initialized.
    :param seed: The desired seed value to be used for variable permutation and other random components of the solver.
    """

    seed = seed % 2147483648  # SCIP seed range.

    # Set up randomization.
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # Disable presolver restarts.
    model.setIntParam('presolving/maxrestarts', 0)


def get_state(model: pyscipopt.scip.Model, obj_norm=None):
    """Extracts the graph representation of the problem at the current solver state.

    The nodes in this graph are the constraints, variables, and cut candidates. Constraints and cuts are connected to
    a variable if and only if this variable appears in the row (cut or constraint).

    :param model: The current model.
    :param obj_norm: The norm of the objective function, provided after the initial calculation to avoid recomputing
        it on every iteration.
    :return: A tuple consisting of the constraint, constraint edge, variable, cut, and cut edge features. The
        constraint, variable, and cuts features are dictionaries of the form {'feature': str, 'values': np.ndarray}. The
        edge features are of the form {'feature': str, 'indices': np.ndarray, 'values': np.ndarray}, where the indices
    """

    if obj_norm is None:
        # Compute the norm of the objective value.
        obj_norm = np.linalg.norm(list(model.getObjective().terms.values()))
        obj_norm = 1 if obj_norm <= 0 else obj_norm

    # Retrieve rows (constraints) and columns (variables).
    rows = model.getLPRowsData()
    cols = model.getLPColsData()
    n_rows = len(rows)
    n_cols = len(cols)

    # Compute the norm of each row.
    row_norms = np.array([row.getNorm() for row in rows])
    row_norms[row_norms == 0] = 1

    # Row (constraint) features.
    row_feats = {}

    # Split constraints of the form lhs <= d^T x <= rhs into two parts (lhs <= d^T x is transformed to -d^T x <= -lhs).
    lhs = np.array([row.getLhs() for row in rows])
    rhs = np.array([row.getRhs() for row in rows])
    has_lhs = [not model.isInfinity(-val) for val in lhs]
    has_rhs = [not model.isInfinity(val) for val in rhs]
    rows = np.array(rows)
    lhs_rows = rows[has_lhs]
    rhs_rows = rows[has_rhs]

    # Compute cosine similarity with the objective function.
    cosines = np.array([get_objCosine(rows[i], row_norms[i], obj_norm) for i in range(n_rows)])
    row_feats['obj_cosine'] = np.concatenate((-cosines[has_lhs], cosines[has_rhs])).reshape(-1, 1)

    # Compute the right-hand side of each constraint.
    row_feats['bias'] = np.concatenate((-(lhs / row_norms)[has_lhs], (rhs / row_norms)[has_rhs])).reshape(-1, 1)

    # Compute tightness indicator.
    row_feats['is_tight'] = np.concatenate(([row.getBasisStatus() == 'lower' for row in lhs_rows],
                                            [row.getBasisStatus() == 'upper' for row in rhs_rows])).reshape(-1, 1)

    duals = np.array([model.getRowDualSol(row) for row in rows]) / (row_norms * obj_norm)
    row_feats['dual'] = np.concatenate((-duals[has_lhs], +duals[has_rhs])).reshape(-1, 1)

    row_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                      row_feats.items()]
    row_feat_names = [n for names in row_feat_names for n in names]
    row_feat_vals = np.concatenate(list(row_feats.values()), axis=-1)
    constraint_feats = {'names': row_feat_names, 'values': row_feat_vals}

    # Constraint edge features.
    # For each row, record a vector [value / row_norm, row_index, column_index] and stack everything into one big
    # matrix (-1x3).
    data = np.array([[rows[i].getVals()[j] / row_norms[i], rows[i].getLPPos(), rows[i].getCols()[j].getLPPos()] for i
                     in range(n_rows) for j in range(len(rows[i].getCols()))])

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
    cons_edge_feats = {'names': edge_feat_names, 'indices': edge_feat_indices, 'values': edge_feat_vals}

    # Column (variable) features.
    col_feats = {}

    # Retrieve column type.
    type_map = {'BINARY': 0, 'INTEGER': 1, 'IMPLINT': 2, 'CONTINUOUS': 3}
    types = np.array([type_map[col.getVar().vtype()] for col in cols])
    col_feats['type'] = np.zeros((n_cols, 4))
    col_feats['type'][np.arange(n_cols), types] = 1

    # Compute normalized column coefficient in the objective function.
    col_feats['obj_coef'] = np.array([col.getObjCoeff() for col in cols]) / obj_norm

    # Get variable lower and upper bounds.
    lb = np.array([col.getLb() for col in cols])
    ub = np.array([col.getUb() for col in cols])
    has_lb = [not model.isInfinity(-val) for val in lb]
    has_ub = [not model.isInfinity(val) for val in ub]
    col_feats['has_lb'] = np.array(has_lb).astype(int)
    col_feats['has_ub'] = np.array(has_ub).astype(int)
    row_feats['is_tight'] = np.concatenate((,
                                            [row.getBasisStatus() == 'upper' for row in rhs_rows])).reshape(-1, 1)
    col_feats['at_lb'] = [col.getBasisStatus() == 'lower' for col in cols[col_feat]]
    col_feats['at_ub'] = s['col']['sol_is_at_ub'].reshape(-1, 1)

    col_feats['sol_frac'] = s['col']['solfracs'].reshape(-1, 1)
    col_feats['sol_frac'][s['col']['types'] == 3] = 0  # continuous have no fractionality
    col_feats['basis_status'] = np.zeros((n_cols, 4))  # LOWER BASIC UPPER ZERO
    col_feats['basis_status'][np.arange(n_cols), s['col']['basestats']] = 1
    col_feats['reduced_cost'] = s['col']['redcosts'].reshape(-1, 1) / obj_norm
    col_feats['age'] = s['col']['ages'].reshape(-1, 1) / (s['stats']['nlps'] + 5)
    col_feats['sol_val'] = s['col']['solvals'].reshape(-1, 1)
    col_feats['inc_val'] = s['col']['incvals'].reshape(-1, 1)
    col_feats['avg_inc_val'] = s['col']['avgincvals'].reshape(-1, 1)

    col_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                      col_feats.items()]
    col_feat_names = [n for names in col_feat_names for n in names]
    col_feat_vals = np.concatenate(list(col_feats.values()), axis=-1)

    var_feats = {'names': col_feat_names, 'values': col_feat_vals, }

    if 'state' not in buffer:
        buffer['state'] = {'obj_norm': obj_norm, 'col_feats': col_feats, 'row_feats': row_feats, 'has_lhs': has_lhs,
                           'has_rhs': has_rhs, 'edge_row_idxs': edge_row_idxs, 'edge_col_idxs': edge_col_idxs,
                           'edge_feats': edge_feats, }

    return constraint_features, edge_features, variable_features


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


def getEdgeData()


def extract_state(model, buffer=None):
    """
    Compute a bipartite graph representation of the solver. In this
    representation, the variables and constraints of the MILP are the
    left- and right-hand side nodes, and an edge links two nodes iff the
    variable is involved in the constraint. Both the nodes and edges carry
    features.
    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    buffer : dict
        A buffer to avoid re-extracting redundant information from the solver
        each time.
    Returns
    -------
    variable_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the variable nodes in the bipartite graph.
    edge_features : dictionary of type ('names': list, 'indices': np.ndarray, 'values': np.ndarray}
        The features associated with the edges in the bipartite graph.
        This is given as a sparse matrix in COO format.
    constraint_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the constraint nodes in the bipartite graph.
    """
    if buffer is None or model.getNNodes() == 1:
        buffer = {}

    # update state from buffer if any
    s = model.getState(buffer['scip_state'] if 'scip_state' in buffer else None)
    buffer['scip_state'] = s

    if 'state' in buffer:
        obj_norm = buffer['state']['obj_norm']
    else:
        obj_norm = np.linalg.norm(s['col']['coefs'])
        obj_norm = 1 if obj_norm <= 0 else obj_norm

    row_norms = s['row']['norms']
    row_norms[row_norms == 0] = 1

    # Column features
    n_cols = len(s['col']['types'])

    if 'state' in buffer:
        col_feats = buffer['state']['col_feats']
    else:
        col_feats = {}
        col_feats['type'] = np.zeros((n_cols, 4))  # BINARY INTEGER IMPLINT CONTINUOUS
        col_feats['type'][np.arange(n_cols), s['col']['types']] = 1
        col_feats['coef_normalized'] = s['col']['coefs'].reshape(-1, 1) / obj_norm

    col_feats['has_lb'] = ~np.isnan(s['col']['lbs']).reshape(-1, 1)
    col_feats['has_ub'] = ~np.isnan(s['col']['ubs']).reshape(-1, 1)
    col_feats['sol_is_at_lb'] = s['col']['sol_is_at_lb'].reshape(-1, 1)
    col_feats['sol_is_at_ub'] = s['col']['sol_is_at_ub'].reshape(-1, 1)
    col_feats['sol_frac'] = s['col']['solfracs'].reshape(-1, 1)
    col_feats['sol_frac'][s['col']['types'] == 3] = 0  # continuous have no fractionality
    col_feats['basis_status'] = np.zeros((n_cols, 4))  # LOWER BASIC UPPER ZERO
    col_feats['basis_status'][np.arange(n_cols), s['col']['basestats']] = 1
    col_feats['reduced_cost'] = s['col']['redcosts'].reshape(-1, 1) / obj_norm
    col_feats['age'] = s['col']['ages'].reshape(-1, 1) / (s['stats']['nlps'] + 5)
    col_feats['sol_val'] = s['col']['solvals'].reshape(-1, 1)
    col_feats['inc_val'] = s['col']['incvals'].reshape(-1, 1)
    col_feats['avg_inc_val'] = s['col']['avgincvals'].reshape(-1, 1)

    col_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                      col_feats.items()]
    col_feat_names = [n for names in col_feat_names for n in names]
    col_feat_vals = np.concatenate(list(col_feats.values()), axis=-1)

    variable_features = {'names': col_feat_names, 'values': col_feat_vals, }

    # Row features

    if 'state' in buffer:
        row_feats = buffer['state']['row_feats']
        has_lhs = buffer['state']['has_lhs']
        has_rhs = buffer['state']['has_rhs']
    else:
        row_feats = {}
        has_lhs = np.nonzero(~np.isnan(s['row']['lhss']))[0]
        has_rhs = np.nonzero(~np.isnan(s['row']['rhss']))[0]
        row_feats['obj_cosine_similarity'] = np.concatenate(
            (-s['row']['objcossims'][has_lhs], +s['row']['objcossims'][has_rhs])).reshape(-1, 1)
        row_feats['bias'] = np.concatenate(
            (-(s['row']['lhss'] / row_norms)[has_lhs], +(s['row']['rhss'] / row_norms)[has_rhs])).reshape(-1, 1)

    row_feats['is_tight'] = np.concatenate((s['row']['is_at_lhs'][has_lhs], s['row']['is_at_rhs'][has_rhs])).reshape(-1,
                                                                                                                     1)

    row_feats['age'] = np.concatenate((s['row']['ages'][has_lhs], s['row']['ages'][has_rhs])).reshape(-1, 1) / (
            s['stats']['nlps'] + 5)

    # # redundant with is_tight
    # tmp = s['row']['basestats']  # LOWER BASIC UPPER ZERO
    # tmp[s['row']['lhss'] == s['row']['rhss']] = 4  # LOWER == UPPER for equality constraints
    # tmp_l = tmp[has_lhs]
    # tmp_l[tmp_l == 2] = 1  # LHS UPPER -> BASIC
    # tmp_l[tmp_l == 4] = 2  # EQU UPPER -> UPPER
    # tmp_l[tmp_l == 0] = 2  # LHS LOWER -> UPPER
    # tmp_r = tmp[has_rhs]
    # tmp_r[tmp_r == 0] = 1  # RHS LOWER -> BASIC
    # tmp_r[tmp_r == 4] = 2  # EQU LOWER -> UPPER
    # tmp = np.concatenate((tmp_l, tmp_r)) - 1  # BASIC UPPER ZERO
    # row_feats['basis_status'] = np.zeros((len(has_lhs) + len(has_rhs), 3))
    # row_feats['basis_status'][np.arange(len(has_lhs) + len(has_rhs)), tmp] = 1

    tmp = s['row']['dualsols'] / (row_norms * obj_norm)
    row_feats['dualsol_val_normalized'] = np.concatenate((-tmp[has_lhs], +tmp[has_rhs])).reshape(-1, 1)

    row_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                      row_feats.items()]
    row_feat_names = [n for names in row_feat_names for n in names]
    row_feat_vals = np.concatenate(list(row_feats.values()), axis=-1)

    constraint_features = {'names': row_feat_names, 'values': row_feat_vals, }

    # Edge features
    if 'state' in buffer:
        edge_row_idxs = buffer['state']['edge_row_idxs']
        edge_col_idxs = buffer['state']['edge_col_idxs']
        edge_feats = buffer['state']['edge_feats']
    else:
        coef_matrix = sp.csr_matrix((s['nzrcoef']['vals'] / row_norms[s['nzrcoef']['rowidxs']],
                                     (s['nzrcoef']['rowidxs'], s['nzrcoef']['colidxs'])),
                                    shape=(len(s['row']['nnzrs']), len(s['col']['types'])))
        coef_matrix = sp.vstack((-coef_matrix[has_lhs, :], coef_matrix[has_rhs, :])).tocoo(copy=False)

        edge_row_idxs, edge_col_idxs = coef_matrix.row, coef_matrix.col
        edge_feats = {}

        edge_feats['coef_normalized'] = coef_matrix.data.reshape(-1, 1)

    edge_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in
                       edge_feats.items()]
    edge_feat_names = [n for names in edge_feat_names for n in names]
    edge_feat_indices = np.vstack([edge_row_idxs, edge_col_idxs])
    edge_feat_vals = np.concatenate(list(edge_feats.values()), axis=-1)

    edge_features = {'names': edge_feat_names, 'indices': edge_feat_indices, 'values': edge_feat_vals, }

    if 'state' not in buffer:
        buffer['state'] = {'obj_norm': obj_norm, 'col_feats': col_feats, 'row_feats': row_feats, 'has_lhs': has_lhs,
                           'has_rhs': has_rhs, 'edge_row_idxs': edge_row_idxs, 'edge_col_idxs': edge_col_idxs,
                           'edge_feats': edge_feats, }

    return constraint_features, edge_features, variable_features


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
