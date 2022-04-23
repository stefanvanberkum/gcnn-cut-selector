"""This module is used to generate problem instances.

Summary
=======
This module provides methods for randomly generated set covering, combinatorial auction, capacitated facility location,
and maximum independent set problem instances. The methods in this module are all based on [1]_.

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
.. [1] Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). *Exact combinatorial optimization with
    graph convolutional neural networks*. Neural Information Processing Systems (NeurIPS 2019), 15580–15592.
    https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
"""

import argparse
import os
from itertools import combinations

import numpy as np
import scipy.sparse


# import utilities


class Graph:
    """Data structure for a generic graph with methods for random graph generation.

    Methods
    =======
    - :meth:`clique_partition`: partitions the graph into cliques using a greedy algorithm.
    - :meth:`erdos_renyi`: Generates an Erdős-Rényi random graph with a given edge probability.
    - :meth:`barabasi_albert`: Generates a Barabási-Albert random graph with a given edge probability.

    :ivar int n_nodes: The number of nodes in the graph.
    :ivar edges: A set of integer tuples representing the edges (i.e., (a, b) denotes an edge between node a and b).
    :ivar degrees: An array of integers denoting the degree of each node in the graph.
    :ivar neighbors: A dictionary that records the neighbors of each node in the graph.
    """

    def __init__(self, n_nodes: int, edges: set[tuple[int, int]], degrees: np.array[int], neighbors: dict[set[int]]):
        self.n_nodes = n_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        """Gets the number of nodes in the graph.

        :return: The number of nodes.
        """

        return self.n_nodes

    def clique_partition(self):
        """Partitions the graph into cliques using a greedy algorithm.

        :return: The resulting clique partition.
        """

        # Sort leftover nodes in descending degree order.
        leftover_nodes = (-self.degrees).argsort().tolist()

        cliques = []
        while leftover_nodes:
            # Pick the leftover node with the largest degree.
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}

            # Get neighbors that are leftover nodes and sort in descending degree order.
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])

            for neighbor in densest_neighbors:
                # Add neighbor to clique if it is a neighbor of all nodes already in the clique.
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]
        return cliques

    @staticmethod
    def erdos_renyi(n_nodes: int, p_edge: float, rng: np.random.RandomState):
        """Generates an Erdős-Rényi random graph with a given edge probability.

        :param n_nodes: The number of nodes in the graph.
        :param p_edge: The probability of generating an edge.
        :param rng: A random number generator.
        :return: The random graph.
        """
        edges = set()
        degrees = np.zeros(n_nodes, dtype=int)
        neighbors = {node: set() for node in range(n_nodes)}
        for edge in combinations(np.arange(n_nodes), 2):
            if rng.uniform() < p_edge:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        return Graph(n_nodes, edges, degrees, neighbors)

    @staticmethod
    def barabasi_albert(n_nodes: int, affinity: int, rng: np.random.RandomState):
        """Generates a Barabási-Albert random graph with a given affinity (number of connections for each new node).

        Based on the algorithm described in [1]_.

        References
        ==========
        .. [1] Barabási, A.-L., & Albert, R. (1999). *Emergence of scaling in random networks*. Science, 286(5439),
            509–512. https://doi.org/10.1126/SCIENCE.286.5439.509

        :param n_nodes: The number of nodes in the graph (m).
        :param affinity: The initial number of nodes, and the number of nodes that each new node will be attached to
            (m = m_0).
        :param rng: A random number generator.
        :return: The random graph.
        """

        assert 1 <= affinity < n_nodes

        edges = set()
        degrees = np.zeros(n_nodes, dtype=int)
        neighbors = {node: set() for node in range(n_nodes)}
        for new_node in range(affinity, n_nodes):
            # First node is connected to all initial nodes.
            if new_node == affinity:
                # Connect first node to all initial nodes.
                neighborhood = np.arange(affinity)
            else:
                # Connect remaining nodes stochastically, using preferential attachment (proportional to degree).
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = rng.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)
        return Graph(n_nodes, edges, degrees, neighbors)


def generate_setcov(n_rows: int, n_cols: int, density: float, filepath: str, rng: np.random.RandomState, max_coef=100):
    """Generates a set covering problem instance and writes to CPLEX LP file.

    Based on the algorithm described in [1]_.

    References
    ==========
    .. [1] Balas, E., & Ho, A. (1980). Set covering algorithms using cutting planes, heuristics, and subgradient
        optimization: A computational study. In M. W. Padberg (Ed.), *Combinatorial Optimization* (pp. 37–60). Springer.
        https://doi.org/10.1007/BFB0120886

    :param n_rows: The desired number of rows.
    :param n_cols: The desired number of columns.
    :param density: The desired density of the constraint matrix (fraction of non-zero elements), in range [0, 1].
    :param filepath: Save file path.
    :param rng: Random number generator.
    :param max_coef: Maximum objective coefficient (>=1).
    :return:
    """

    n_nonzero = int(n_rows * n_cols * density)

    assert n_nonzero >= n_rows  # At least one non-zero entry per row.
    assert n_nonzero >= 2 * n_cols  # At least two non-zero entries per column.

    indices = rng.choice(n_cols, size=n_nonzero)  # Assign non-zero entries to random columns.
    indices[:2 * n_cols] = np.repeat(np.arange(n_cols), 2)  # Force at least two non-zero entries per column,
    _, col_nonzero = np.unique(indices, return_counts=True)  # Compute the number of non-zero entries for each column.

    # Force at least one non-zero entry per row.
    indices[:n_rows] = rng.permutation(n_rows)

    i = 0
    indptr = [0]  # See scipy.sparse.csc_matrix documentation for name origin.
    for count in col_nonzero:
        # For each column, we assign the remaining non-zero elements to random rows (besides the forced first n_rows).
        # Sampling is done without replacement, because each row can appear only once in each column.
        if i >= n_rows:
            # Assign all non-zero elements to random rows.
            indices[i:i + count] = rng.choice(n_rows, size=count, replace=False)
        elif i + count > n_rows:
            # Assign all non-zero elements that are not yet fixed to rows that have not yet been chosen for this column.
            remaining_rows = np.setdiff1d(np.arange(n_rows), indices[i:n_rows], assume_unique=True)
            indices[n_rows:i + count] = rng.choice(remaining_rows, size=i + count - n_rows, replace=False)
        i += n
        indptr.append(i)

    # Draw objective coefficients from {1, 2, ..., max_coef}.
    c = rng.randint(max_coef, size=n_cols) + 1

    # Sparce CSC to sparse CSR matrix.
    matrix = scipy.sparse.csc_matrix((np.ones(len(indices), dtype=int), indices, indptr),
                                     shape=(n_rows, n_cols)).tocsr()
    indices = matrix.indices
    indptr = matrix.indptr

    # Write problem to CPLEX LP file.
    with open(filepath, 'w') as file:
        file.write("minimize\nOBJ:")
        file.write("".join([f" +{c[j]} x{j + 1}" for j in range(n_cols)]))

        file.write("\n\nsubject to\n")
        for i in range(n_rows):
            row_cols_str = "".join([f" +1 x{j + 1}" for j in indices[indptr[i]:indptr[i + 1]]])
            file.write(f"C{i}:" + row_cols_str + f" >= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" x{j + 1}" for j in range(n_cols)]))


def generate_combauc(random, filename, n_items=100, n_bids=500, min_value=1, max_value=100, value_deviation=0.5,
                     add_item_prob=0.7, max_n_sub_bids=5, additivity=0.2, budget_factor=1.5, resale_factor=0.5,
                     integers=False, warnings=False):
    """
    Generate a Combinatorial Auction problem following the 'arbitrary' scheme found in section 4.3. of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.
    Saves it as a CPLEX LP file.
    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_items : int
        The number of items.
    n_bids : int
        The number of bids.
    min_value : int
        The minimum resale value for an item.
    max_value : int
        The maximum resale value for an item.
    value_deviation : int
        The deviation allowed for each bidder's private value of an item, relative from max_value.
    add_item_prob : float in [0, 1]
        The probability of adding a new item to an existing bundle.
    max_n_sub_bids : int
        The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
    additivity : float
        Additivity parameter for bundle prices. Note that additivity < 0 gives sub-additive bids, while additivity >
        0 gives super-additive bids.
    budget_factor : float
        The budget factor for each bidder, relative to their initial bid's price.
    resale_factor : float
        The resale factor for each bidder, relative to their initial bid's resale value.
    integers : logical
        Should bid's prices be integral ?
    warnings : logical
        Should warnings be printed ?
    """

    assert min_value >= 0 and max_value >= min_value
    assert add_item_prob >= 0 and add_item_prob <= 1

    def choose_next_item(bundle_mask, interests, compats, add_item_prob, random):
        n_items = len(interests)
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return random.choice(n_items, p=prob)

    # common item values (resale price)
    values = min_value + (max_value - min_value) * random.rand(n_items)

    # item compatibilities
    compats = np.triu(random.rand(n_items, n_items), k=1)
    compats = compats + compats.transpose()
    compats = compats / compats.sum(1)

    bids = []
    n_dummy_items = 0

    # create bids, one bidder at a time
    while len(bids) < n_bids:

        # bidder item values (buy price) and interests
        private_interests = random.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # substitutable bids of this bidder
        bidder_bids = {}

        # generate initial bundle, choose first item according to bidder interests
        prob = private_interests / private_interests.sum()
        item = random.choice(n_items, p=prob)
        bundle_mask = np.full(n_items, 0)
        bundle_mask[item] = 1

        # add additional items, according to bidder interests and item compatibilities
        while random.rand() < add_item_prob:
            # stop when bundle full (no item left)
            if bundle_mask.sum() == n_items:
                break
            item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
            bundle_mask[item] = 1

        bundle = np.nonzero(bundle_mask)[0]

        # compute bundle price with value additivity
        price = private_values[bundle].sum() + np.power(len(bundle), 1 + additivity)
        if integers:
            price = int(price)

        # drop negativaly priced bundles
        if price < 0:
            if warnings:
                print("warning: negatively priced bundle avoided")
            continue

        # bid on initial bundle
        bidder_bids[frozenset(bundle)] = price

        # generate candidates substitutable bundles
        sub_candidates = []
        for item in bundle:

            # at least one item must be shared with initial bundle
            bundle_mask = np.full(n_items, 0)
            bundle_mask[item] = 1

            # add additional items, according to bidder interests and item compatibilities
            while bundle_mask.sum() < len(bundle):
                item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, random)
                bundle_mask[item] = 1

            sub_bundle = np.nonzero(bundle_mask)[0]

            # compute bundle price with value additivity
            sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            if integers:
                sub_price = int(sub_price)

            sub_candidates.append((sub_bundle, sub_price))

        # filter valid candidates, higher priced candidates first
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        for bundle, price in [sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break

            if price < 0:
                if warnings:
                    print("warning: negatively priced substitutable bundle avoided")
                continue

            if price > budget:
                if warnings:
                    print("warning: over priced substitutable bundle avoided")
                continue

            if values[bundle].sum() < min_resale_value:
                if warnings:
                    print("warning: substitutable bundle below min resale value avoided")
                continue

            if frozenset(bundle) in bidder_bids:
                if warnings:
                    print("warning: duplicated substitutable bundle avoided")
                continue

            bidder_bids[frozenset(bundle)] = price

        # add XOR constraint if needed (dummy item)
        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        # place bids
        for bundle, price in bidder_bids.items():
            bids.append((list(bundle) + dummy_item, price))

    # generate the LP file
    with open(filename, 'w') as file:
        bids_per_item = [[] for item in range(n_items + n_dummy_items)]

        file.write("maximize\nOBJ:")
        for i, bid in enumerate(bids):
            bundle, price = bid
            file.write(f" +{price} x{i + 1}")
            for item in bundle:
                bids_per_item[item].append(i)

        file.write("\n\nsubject to\n")
        for item_bids in bids_per_item:
            if item_bids:
                for i in item_bids:
                    file.write(f" +1 x{i + 1}")
                file.write(f" <= 1\n")

        file.write("\nbinary\n")
        for i in range(len(bids)):
            file.write(f" x{i + 1}")


def generate_capfac(random, filename, n_customers, n_facilities, ratio):
    """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.
    Saves it as a CPLEX LP file.
    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """
    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    demands = rng.randint(5, 35 + 1, size=n_customers)
    capacities = rng.randint(10, 160 + 1, size=n_facilities)
    fixed_costs = rng.randint(100, 110 + 1, size=n_facilities) * np.sqrt(capacities) + rng.randint(90 + 1,
                                                                                                   size=n_facilities)
    fixed_costs = fixed_costs.astype(int)

    total_demand = demands.sum()
    total_capacity = capacities.sum()

    # adjust capacities according to ratio
    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)
    total_capacity = capacities.sum()

    # transportation costs
    trans_costs = np.sqrt((c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 + (
            c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nobj:")
        file.write("".join(
            [f" +{trans_costs[i, j]} x_{i + 1}_{j + 1}" for i in range(n_customers) for j in range(n_facilities)]))
        file.write("".join([f" +{fixed_costs[j]} y_{j + 1}" for j in range(n_facilities)]))

        file.write("\n\nsubject to\n")
        for i in range(n_customers):
            file.write(
                f"demand_{i + 1}:" + "".join([f" -1 x_{i + 1}_{j + 1}" for j in range(n_facilities)]) + f" <= -1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j + 1}:" + "".join([f" +{demands[i]} x_{i + 1}_{j + 1}" for i in
                                                       range(n_customers)]) + f" -{capacities[j]} y_{j + 1} <= 0\n")

        # optional constraints for LP relaxation tightening
        file.write("total_capacity:" + "".join(
            [f" -{capacities[j]} y_{j + 1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"affectation_{i + 1}_{j + 1}: +1 x_{i + 1}_{j + 1} -1 y_{j + 1} <= 0")

        file.write("\nbounds\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"0 <= x_{i + 1}_{j + 1} <= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" y_{j + 1}" for j in range(n_facilities)]))


def generate_indset(graph, filepath):
    """Generates a maximum independent set problem instance and writes to CPLEX LP file.

    :param graph: The graph from which to build the independent set problem.
    :param filepath: Save file path.
    """

    cliques = graph.clique_partition()
    inequalities = set(graph.edges)
    for clique in cliques:
        clique = tuple(sorted(clique))
        for edge in combinations(clique, 2):
            inequalities.remove(edge)
        if len(clique) > 1:
            inequalities.add(clique)

    # Put trivial inequalities for nodes that didn't appear
    # in the constraints, otherwise SCIP will complain
    used_nodes = set()
    for group in inequalities:
        used_nodes.update(group)
    for node in range(10):
        if node not in used_nodes:
            inequalities.add((node,))

    with open(filename, 'w') as lp_file:
        lp_file.write("maximize\nOBJ:" + "".join([f" + 1 x{node + 1}" for node in range(len(graph))]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, group in enumerate(inequalities):
            lp_file.write(f"C{count + 1}:" + "".join([f" + x{node + 1}" for node in sorted(group)]) + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node + 1}" for node in range(len(graph))]) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', help='MILP instance type to process.',
                        choices=['setcover', 'cauctions', 'facilities', 'indset'], )
    # parser.add_argument('-s', '--seed', help='Random generator seed (default 0).', type=utilities.valid_seed,
    #                    default=0, )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    if args.problem == 'setcover':
        nrows = 500
        ncols = 1000
        dens = 0.05
        max_coef = 100

        filenames = []
        nrowss = []
        ncolss = []
        denss = []

        # train instances
        n = 10000
        lp_dir = f'data/instances/setcover/train_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # validation instances
        n = 2000
        lp_dir = f'data/instances/setcover/valid_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # small transfer instances
        n = 100
        nrows = 500
        lp_dir = f'data/instances/setcover/transfer_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # medium transfer instances
        n = 100
        nrows = 1000
        lp_dir = f'data/instances/setcover/transfer_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # big transfer instances
        n = 100
        nrows = 2000
        lp_dir = f'data/instances/setcover/transfer_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # test instances
        n = 2000
        nrows = 500
        ncols = 1000
        lp_dir = f'data/instances/setcover/test_{nrows}r_{ncols}c_{dens}d'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nrowss.extend([nrows] * n)
        ncolss.extend([ncols] * n)
        denss.extend([dens] * n)

        # actually generate the instances
        for filename, nrows, ncols, dens in zip(filenames, nrowss, ncolss, denss):
            print(f'  generating file {filename} ...')
            generate_setcov(nrows=nrows, ncols=ncols, density=dens, filename=filename, rng=rng, max_coef=max_coef)

        print('done.')

    elif args.problem == 'indset':
        number_of_nodes = 500
        affinity = 4

        filenames = []
        nnodess = []

        # train instances
        n = 10000
        lp_dir = f'data/instances/indset/train_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # validation instances
        n = 2000
        lp_dir = f'data/instances/indset/valid_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # small transfer instances
        n = 100
        number_of_nodes = 500
        lp_dir = f'data/instances/indset/transfer_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # medium transfer instances
        n = 100
        number_of_nodes = 1000
        lp_dir = f'data/instances/indset/transfer_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # big transfer instances
        n = 100
        number_of_nodes = 1500
        lp_dir = f'data/instances/indset/transfer_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # test instances
        n = 2000
        number_of_nodes = 500
        lp_dir = f'data/instances/indset/test_{number_of_nodes}_{affinity}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nnodess.extend([number_of_nodes] * n)

        # actually generate the instances
        for filename, nnodes in zip(filenames, nnodess):
            print(f"  generating file {filename} ...")
            graph = Graph.barabasi_albert(nnodes, affinity, rng)
            generate_indset(graph, filename)

        print("done.")

    elif args.problem == 'cauctions':
        number_of_items = 100
        number_of_bids = 500
        filenames = []
        nitemss = []
        nbidss = []

        # train instances
        n = 10000
        lp_dir = f'data/instances/cauctions/train_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # validation instances
        n = 2000
        lp_dir = f'data/instances/cauctions/valid_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # small transfer instances
        n = 100
        number_of_items = 100
        number_of_bids = 500
        lp_dir = f'data/instances/cauctions/transfer_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # medium transfer instances
        n = 100
        number_of_items = 200
        number_of_bids = 1000
        lp_dir = f'data/instances/cauctions/transfer_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # big transfer instances
        n = 100
        number_of_items = 300
        number_of_bids = 1500
        lp_dir = f'data/instances/cauctions/transfer_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # test instances
        n = 2000
        number_of_items = 100
        number_of_bids = 500
        lp_dir = f'data/instances/cauctions/test_{number_of_items}_{number_of_bids}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        nitemss.extend([number_of_items] * n)
        nbidss.extend([number_of_bids] * n)

        # actually generate the instances
        for filename, nitems, nbids in zip(filenames, nitemss, nbidss):
            print(f"  generating file {filename} ...")
            generate_combauc(rng, filename, n_items=nitems, n_bids=nbids, add_item_prob=0.7)

        print("done.")

    elif args.problem == 'facilities':
        number_of_customers = 100
        number_of_facilities = 100
        ratio = 5
        filenames = []
        ncustomerss = []
        nfacilitiess = []
        ratios = []

        # train instances
        n = 10000
        lp_dir = f'data/instances/facilities/train_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # validation instances
        n = 2000
        lp_dir = f'data/instances/facilities/valid_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # small transfer instances
        n = 100
        number_of_customers = 100
        number_of_facilities = 100
        lp_dir = f'data/instances/facilities/transfer_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # medium transfer instances
        n = 100
        number_of_customers = 200
        lp_dir = f'data/instances/facilities/transfer_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # big transfer instances
        n = 100
        number_of_customers = 400
        lp_dir = f'data/instances/facilities/transfer_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # test instances
        n = 2000
        number_of_customers = 100
        number_of_facilities = 100
        lp_dir = f'data/instances/facilities/test_{number_of_customers}_{number_of_facilities}_{ratio}'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        filenames.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nfacilitiess.extend([number_of_facilities] * n)
        ratios.extend([ratio] * n)

        # actually generate the instances
        for filename, ncs, nfs, r in zip(filenames, ncustomerss, nfacilitiess, ratios):
            print(f"  generating file {filename} ...")
            generate_capfac(rng, filename, n_customers=ncs, n_facilities=nfs, ratio=r)

        print("done.")
