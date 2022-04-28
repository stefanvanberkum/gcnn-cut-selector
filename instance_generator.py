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

import os
from itertools import combinations

import numpy as np
import scipy.sparse


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

    def __init__(self, n_nodes: int, edges: set[tuple[int, int]], degrees: np.array, neighbors: dict[set[int]]):
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

        A clique is a set of nodes for which any two nodes in the set are connected.

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
    def barabasi_albert(n_nodes: int, rng: np.random.RandomState, affinity=4):
        """Generates a Barabási-Albert random graph with a given affinity (number of connections for each new node).

        This method dynamically models a scale-free network, based on the algorithm described in [1]_. We start with
        an initial number of nodes :math:`m_0`, and then at every iteration a new node is added and connected to
        :math:`m_0` other nodes, until the desired number of nodes is reached. Note that this implies that the first
        node that we add is connected to all initial nodes. Nodes are randomly connected to other nodes,
        with a probability that is proportional to the connectivity of the other node.

        References
        ==========
        .. [1] Barabási, A.-L., & Albert, R. (1999). *Emergence of scaling in random networks*. Science, 286(5439),
            509–512. https://doi.org/10.1126/SCIENCE.286.5439.509

        :param n_nodes: The number of nodes in the graph (m).
        :param rng: A random number generator.
        :param affinity: The initial number of nodes, and the number of nodes that each new node will be attached to
            (m = m_0).
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


def generate_instances():
    """Generates set covering, combinatorial auction, capacitated facility, and independent set problem instances in
    accordance with our data generation scheme.
    """
    rng = np.random.RandomState(0)
    generate_setcovs(rng)
    generate_combaucs(rng)
    generate_capfacs(rng)
    generate_indsets(rng)


def generate_setcovs(rng: np.random.RandomState):
    """Generates set covering problem instances in accordance with our data generation scheme.

    This method generates 10000 instances for training, 2000 for validation, and another 2000 for testing (500x1000).
    Moreover, it generates three set of 100 instances each for evaluation, with dimensions 500x1000 (Easy),
    1000x1000 (Medium), and 2000x1000 (Hard).

    :param rng: A random number generator.
    """

    filepaths = []
    rows = []

    # Training instances (500x1000).
    n_instances = 10000
    n_rows = 500
    lp_dir = f'data/instances/setcover/train_{n_rows}r'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    rows.extend([n_rows] * n_instances)

    # Validation instances (500x1000).
    n_instances = 2000
    n_rows = 500
    lp_dir = f'data/instances/setcover/valid_{n_rows}r'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    rows.extend([n_rows] * n_instances)

    # Testing instances (500x1000).
    n_instances = 2000
    n_rows = 500
    lp_dir = f'data/instances/setcover/test_{n_rows}r'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    rows.extend([n_rows] * n_instances)

    # Easy evaluation instances (500x1000).
    n_instances = 100
    n_rows = 500
    lp_dir = f'data/instances/setcover/eval_{n_rows}r'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    rows.extend([n_rows] * n_instances)

    # Medium evaluation instances (1000x1000).
    n_instances = 100
    n_rows = 1000
    lp_dir = f'data/instances/setcover/eval_{n_rows}r'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    rows.extend([n_rows] * n_instances)

    # Hard evaluation instances (2000x1000).
    n_instances = 100
    n_rows = 2000
    lp_dir = f'data/instances/setcover/eval_{n_rows}r'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    rows.extend([n_rows] * n_instances)

    # Actually generate the instances.
    for filepath, n_rows in zip(filepaths, rows):
        generate_setcov(n_rows, filepath, rng)


def generate_setcov(n_rows: int, filepath: str, rng: np.random.RandomState, n_cols=1000, density=0.05, max_coef=100):
    """Generates a set covering problem instance and writes it to a CPLEX LP file.

    This method is based on the algorithm described in [1]_, and randomly creates a coefficient matrix with the
    desired density, subject to the requirement that every row has at least two non-zero entries, and each column has
    at least one.

    References
    ==========
    .. [1] Balas, E., & Ho, A. (1980). Set covering algorithms using cutting planes, heuristics, and subgradient
        optimization: A computational study. In M. W. Padberg (Ed.), *Combinatorial Optimization* (pp. 37–60). Springer.
        https://doi.org/10.1007/BFB0120886

    :param n_rows: The desired number of rows.
    :param filepath: The desired save file path.
    :param rng: A random number generator.
    :param n_cols: The desired number of columns.
    :param density: The desired density of the constraint matrix (fraction of non-zero elements), in interval [0, 1].
    :param max_coef: The maximum objective coefficient (>=1).
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
        i += count
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


def generate_combaucs(rng: np.random.RandomState):
    """Generates combinatorial auction problem instances in accordance with our data generation scheme.

    This method generates 10000 instances for training, 2000 for validation, and another 2000 for testing (100x500).
    Moreover, it generates three set of 100 instances each for evaluation, with dimensions 100x500 (Easy),
    200x1000 (Medium), and 300x1500 (Hard).

    :param rng: A random number generator.
    """

    filepaths = []
    items = []
    bids = []

    # Training instances (100x500).
    n_instances = 10000
    n_items = 100
    n_bids = 500
    lp_dir = f'data/instances/cauctions/train_{n_items}i_{n_bids}b'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    items.extend([n_items] * n_instances)
    bids.extend([n_bids] * n_instances)

    # Validation instances (100x500).
    n_instances = 2000
    n_items = 100
    n_bids = 500
    lp_dir = f'data/instances/cauctions/valid_{n_items}i_{n_bids}b'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    items.extend([n_items] * n_instances)
    bids.extend([n_bids] * n_instances)

    # Test instances (100x500).
    n_instances = 2000
    n_items = 100
    n_bids = 500
    lp_dir = f'data/instances/cauctions/test_{n_items}i_{n_bids}b'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    items.extend([n_items] * n_instances)
    bids.extend([n_bids] * n_instances)

    # Easy evaluation instances (100x500).
    n_instances = 100
    n_items = 100
    n_bids = 500
    lp_dir = f'data/instances/cauctions/eval_{n_items}i_{n_bids}b'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    items.extend([n_items] * n_instances)
    bids.extend([n_bids] * n_instances)

    # Medium evaluation instances (200x1000).
    n_instances = 100
    n_items = 200
    n_bids = 1000
    lp_dir = f'data/instances/cauctions/eval_{n_items}i_{n_bids}b'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    items.extend([n_items] * n_instances)
    bids.extend([n_bids] * n_instances)

    # Hard evaluation instances (300x1500).
    n_instances = 100
    n_items = 300
    n_bids = 1500
    lp_dir = f'data/instances/cauctions/eval_{n_items}i_{n_bids}b'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    items.extend([n_items] * n_instances)
    bids.extend([n_bids] * n_instances)

    # Actually generate the instances.
    for n_items, n_bids, filepath in zip(items, bids, filepaths):
        generate_combauc(n_items, n_bids, filepath, rng)


def generate_combauc(n_items: int, n_bids: int, filepath: str, rng: np.random.RandomState, min_value=1, max_value=100,
                     max_deviation=0.5, add_prob=0.7, max_sub_bids=5, additivity=0.2, budget_factor=1.5,
                     resale_factor=0.5, integers=False):
    """Generates a combinatorial auction problem instance and writes it to a CPLEX LP file.

    This method iteratively generates bids, based on the algorithm described in Section 4.3 of [1]_. First,
    we generate common resale values for each good from the interval [*min_value, max_value*). Then, we generate
    compatibilities between goods (the prior probability that they appear together in a bundle). Then,
    for each bidder we randomly generate the deviation from the common resale price by multiplying *max_value *
    max_deviation* by the bidder interests that are randomly generated from :math:`U(0, 1)`. Then, an initial bundle
    is constructed by first taking a random item with probability proportional to the bidder interests. We now keep
    randomly adding items based on add_prob with probabilities proportional to the item compatibility with items
    already present in the bundle, and the bidder's private valuation of the item. The bid for this bundle is then
    computed by adding an additivity term :math:`n^{(1 + additivity)}` to the bidder's private valuation of the
    items, where *n* denotes the size of the bundle. Finally, we generate a maximum of *max_sub_bids* bids that are
    substitutable for the original bid, provided that each of them requests at least one good from the original bid.
    Bundles are substitutable if they fall within the bidder's budget (*price * budget_factor*) and above the minimum
    resale value (*price * resale_factor*), where *price* denotes the bidder's initial bid.

    References
    ==========
    .. [1] Leyton-Brown, K., Pearson, M., & Shoham, Y. (2000). Towards a universal test suite for combinatorial
        auction algorithms. *Electronic Commerce (EC 2000)*, 66–76. https://doi.org/10.1145/352871

    :param n_items: The desired number of items in auction.
    :param n_bids: The desired number of bids.
    :param filepath: The desired save file path.
    :param rng: A random number generator.
    :param min_value: The minimum resale value for an item.
    :param max_value: The maximum resale value for an item.
    :param max_deviation: The amount each bidder's private valuation of an item is allowed to deviate relative to
        max_value.
    :param add_prob: The probability of adding another item to a bundle on each iteration.
    :param max_sub_bids: The maximum number of substitutable bids.
    :param additivity: Additivity parameter for bundle prices, values >0 means that goods are complements and values
        <0 implies that goods are substitutes.
    :param budget_factor: The budget factor for each bidder, relative to their initial bid's price.
    :param resale_factor: The resale factor for each bidder, relative to their initial bid's resale value.
    :param integers: True if bid prices should be integer.
    """

    assert 0 <= min_value <= max_value
    assert 0 <= add_prob <= 1

    def choose_next(chosen: np.array, compat: np.array, interests: np.array, random: np.random.RandomState):
        """Choose a next item with probabilities proportional to item compatibility with chosen items and a bidder's
        interests.

        :param chosen: A binary array where element i equals one of that item has been chosen already.
        :param compat: Item compatibilities.
        :param interests: The bidder's interests, an array of random numbers on the interval [0, 1).
        :param random: A random number generator.
        :return: The index of the chosen item.
        """
        items = len(interests)
        p = (1 - chosen) * compat[chosen, :].mean(axis=0) * interests
        p /= p.sum()
        return random.choice(items, p=p)

    # Randomly generate common resale values for each good between the minimum and maximum value.
    values = min_value + (max_value - min_value) * rng.rand(n_items)

    # Randomly generate item compatibilities (how likely goods are to appear together in a bundle).
    compats = np.triu(rng.rand(n_items, n_items), k=1)  # Generate random upper triangle with zero diagonal.
    compats = compats + compats.transpose()  # Make it symmetric (as compatibilities are symmetric).
    compats = compats / compats.sum(1)  # Scale each column by the sum of its elements to form a probability.

    bids = []
    n_dummy_items = 0

    # Iteratively generate bids, one bidder at a time.
    while len(bids) < n_bids:
        # Add random deviations in interval [-max_value * max_deviation, max_value * max_deviation).
        bidder_interests = rng.rand(n_items)
        private_values = values + max_value * max_deviation * (2 * bidder_interests - 1)

        # Generate an initial bundle, choosing first item according to bidder interests.
        prob = bidder_interests / bidder_interests.sum()
        item = rng.choice(n_items, p=prob)
        chosen_items = np.full(n_items, 0)
        chosen_items[item] = 1

        # Add additional items according to item compatibilities and bidder interests.
        while rng.rand() < add_prob:
            # Stop when there are no items left to choose from.
            if chosen_items.sum() == n_items:
                break
            item = choose_next(chosen_items, compats, bidder_interests, rng)
            chosen_items[item] = 1
        bundle = np.nonzero(chosen_items)[0]

        # Compute bundle bid using value additivity.
        price = private_values[bundle].sum() + np.power(len(bundle), 1 + additivity)
        if integers:
            price = int(price)

        # Drop negatively priced bundles.
        if price < 0:
            continue

        # Record bid on initial bundle.
        bidder_bids = {frozenset(bundle): price}

        # Generate candidate substitutable bundles.
        sub_candidates = []
        for item in bundle:
            # Enforce that at least one item must be shared with initial bundle.
            chosen_items = np.full(n_items, 0)
            chosen_items[item] = 1

            # Add additional items according to item compatibilities and bidder interests.
            while chosen_items.sum() < len(bundle):
                item = choose_next(chosen_items, compats, bidder_interests, rng)
                chosen_items[item] = 1
            sub_bundle = np.nonzero(chosen_items)[0]

            # Compute bundle bid using value additivity.
            sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            if integers:
                sub_price = int(sub_price)
            sub_candidates.append((sub_bundle, sub_price))

        # Filter valid candidates, evaluating higher priced candidates first.
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        for bundle, price in [sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

            # Stop when the maximum number of substitutable bids is reached, or the number of bids in general.
            if len(bidder_bids) >= max_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break

            # Drop negatively priced bundles.
            if price < 0:
                continue

            # Drop bundles that the bidder cannot afford.
            if price > budget:
                continue

            # Drop bundles that have a too low resale value.
            if values[bundle].sum() < min_resale_value:
                continue

            # Drop duplicate bundles.
            if frozenset(bundle) in bidder_bids:
                continue

            bidder_bids[frozenset(bundle)] = price

        # Enforce XOR nature of substitute bids by including a dummy item.
        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        # Record bids.
        for bundle, price in bidder_bids.items():
            bids.append((list(bundle) + dummy_item, price))

    # Write problem to CPLEX LP file.
    with open(filepath, 'w') as file:
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


def generate_capfacs(rng: np.random.RandomState):
    """Generates capacitated facility problem instances in accordance with our data generation scheme.

    This method generates 10000 instances for training, 2000 for validation, and another 2000 for testing (100x100).
    Moreover, it generates three set of 100 instances each for evaluation, with dimensions 100x100 (Easy),
    100x200 (Medium), and 100x400 (Hard).

    :param rng: A random number generator.
    """

    filepaths = []
    customers = []

    # Training instances (100x100).
    n_instances = 10000
    n_customers = 100
    lp_dir = f'data/instances/facilities/train_{n_customers}c'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    customers.extend([n_customers] * n_instances)

    # Validation instances (100x100).
    n_instances = 2000
    n_customers = 100
    lp_dir = f'data/instances/facilities/valid_{n_customers}c'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    customers.extend([n_customers] * n_instances)

    # Testing instances (100x100).
    n_instances = 2000
    n_customers = 100
    lp_dir = f'data/instances/facilities/test_{n_customers}c'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    customers.extend([n_customers] * n_instances)

    # Easy evaluation instances (100x100).
    n_instances = 100
    n_customers = 100
    lp_dir = f'data/instances/facilities/eval_{n_customers}c'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    customers.extend([n_customers] * n_instances)

    # Medium evaluation instances (100x200).
    n_instances = 100
    n_customers = 200
    lp_dir = f'data/instances/facilities/eval_{n_customers}c'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    customers.extend([n_customers] * n_instances)

    # Hard evaluation instances (100x400).
    n_instances = 100
    n_customers = 400
    lp_dir = f'data/instances/facilities/eval_{n_customers}c'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    customers.extend([n_customers] * n_instances)

    # Actually generate the instances.
    for n_customers, filepath in zip(customers, filepaths):
        generate_capfac(n_customers, filepath, rng)


def generate_capfac(n_customers, filepath, rng, n_facilities=100, ratio=5):
    """Generates a capacitated facility problem instance and writes it to a CPLEX LP file.

    This method randomly generates costs, capacities, demands, based on the algorithm described in [1]_. For this
    purpose, we first randomly place facilities and customers on a 1x1 surface and multiply the Euclidian distance
    between them by 10 to obtain unit cost of serving a customer from a particular facility. Then, demands are generated
    from :math:`U(5, 35)`. Capacities :math:`s_j` are generated from :math:`U(10, 160)` and the fixed costs are
    using the formula :math:`U(0, 90) + U(100, 110) \cdot \sqrt{s_j}`, to reflect economies of scale. Finally,
    capacities are scaled to the desired capacity to demand ratio.

    References
    ==========
    .. [1] Cornuejols, G., Sridharan, R., & Thizy, J. M. (1991). A comparison of heuristics and relaxations for the
        capacitated plant location problem. *European Journal of Operational Research*, 50(3), 280–297.
        https://doi.org/10.1016/0377-2217(91)90261-S

    :param n_customers: The desired number of customers.
    :param filepath: The desired save file path.
    :param rng: A random number generator.
    :param n_facilities: The desired number of facilities.
    :param ratio: The desired capacity to demand ratio.
    """

    # Randomly place facilities on a 1x1 surface.
    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    # Randomly place customers on a 1x1 surface.
    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    # Generate demand, capacities, and fixed costs.
    demand = rng.randint(5, 35 + 1, size=n_customers)
    capacities = rng.randint(10, 160 + 1, size=n_facilities)
    fixed_costs = rng.randint(90 + 1, size=n_facilities) + rng.randint(100, 110 + 1, size=n_facilities)
    fixed_costs = fixed_costs.astype(int)

    total_demand = demand.sum()
    total_capacity = capacities.sum()

    # Adjust capacities according to desired capacity to demand ratio.
    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)

    # Compute (total) transportation costs.
    trans_costs = np.sqrt((c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 + (
            c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demand.reshape((-1, 1))

    # Write problem to CPLEX LP file.
    with open(filepath, 'w') as file:
        file.write("minimize\nobj:")
        file.write("".join(
            [f" +{trans_costs[i, j]} x_{i + 1}_{j + 1}" for i in range(n_customers) for j in range(n_facilities)]))
        file.write("".join([f" +{fixed_costs[j]} y_{j + 1}" for j in range(n_facilities)]))

        file.write("\n\nsubject to\n")
        for i in range(n_customers):
            file.write(
                f"demand_{i + 1}:" + "".join([f" -1 x_{i + 1}_{j + 1}" for j in range(n_facilities)]) + f" <= -1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j + 1}:" + "".join([f" +{demand[i]} x_{i + 1}_{j + 1}" for i in
                                                       range(n_customers)]) + f" -{capacities[j]} y_{j + 1} <= 0\n")

        # Optional constraints for LP relaxation tightening.
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


def generate_indsets(rng: np.random.RandomState):
    """Generates independent set problem instances in accordance with our data generation scheme.

    This method generates 10000 instances for training, 2000 for validation, and another 2000 for testing (500 nodes).
    Moreover, it generates three set of 100 instances each for evaluation, with 500 (Easy), 1000 (Medium),
    and 1500 (Hard) nodes.

    :param rng: A random number generator.
    """

    filepaths = []
    nodes = []

    # Training instances (500 nodes).
    n_instances = 10000
    n_nodes = 500
    lp_dir = f'data/instances/indset/train_{n_nodes}n'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    nodes.extend([n_nodes] * n_instances)

    # Validation instances (500 nodes).
    n_instances = 2000
    n_nodes = 500
    lp_dir = f'data/instances/indset/valid_{n_nodes}n'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    nodes.extend([n_nodes] * n_instances)

    # Testing instances (500 nodes).
    n_instances = 2000
    n_nodes = 500
    lp_dir = f'data/instances/indset/test_{n_nodes}n'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    nodes.extend([n_nodes] * n_instances)

    # Easy evaluation instances (500 nodes).
    n_instances = 100
    n_nodes = 500
    lp_dir = f'data/instances/indset/eval_{n_nodes}n'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    nodes.extend([n_nodes] * n_instances)

    # Medium evaluation instances (1000 nodes).
    n_instances = 100
    n_nodes = 1000
    lp_dir = f'data/instances/indset/eval_{n_nodes}n'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    nodes.extend([n_nodes] * n_instances)

    # Hard evaluation instances (1500 nodes).
    n_instances = 100
    n_nodes = 1500
    lp_dir = f'data/instances/indset/eval_{n_nodes}n'
    os.makedirs(lp_dir)
    filepaths.extend([os.path.join(lp_dir, f'instance_{i + 1}.lp') for i in range(n_instances)])
    nodes.extend([n_nodes] * n_instances)

    # Actually generate the instances.
    for n_nodes, filepath in zip(nodes, filepaths):
        graph = Graph.barabasi_albert(n_nodes, rng)
        generate_indset(graph, filepath)


def generate_indset(graph, filepath):
    """Generates a maximum independent set problem instance and writes it to a CPLEX LP file.

    This method generates the maximum independent set problem using a previously generated graph, based on the
    algorithm described in [ 1]_. For this purpose, we start by noting that one can only select a single node from
    any two nodes that are connected by an edge. While this is a valid formulation for our problem, we can strengthen
    it by noting that if we have a clique *S* (a fully-connected set), any independent set can only pick at most one
    node from *S*. For each clique, we can therefore add the constraint that we may only pick a single node from it,
    and remove all previously added constraints for edges in the clique.

    References
    ==========
    .. [1] Bergman, D., Cire, A. A., Van Hoeve, W.-J., & Hooker, J. (2016). *Decision diagrams for optimization*.
        Springer. https://doi.org/10.1007/978-3-319-42849-9

    :param graph: The graph from which to build the independent set problem.
    :param filepath: The desired save file path.
    """

    # Partition graph into cliques.
    cliques = graph.clique_partition()

    inequalities = set(graph.edges)
    for clique in cliques:
        clique = tuple(sorted(clique))

        # Remove all redundant edge constraints.
        for edge in combinations(clique, 2):
            inequalities.remove(edge)
        if len(clique) > 1:
            # Add an inequality specifying we can only select a single node from the clique.
            inequalities.add(clique)

    # Put trivial inequalities for nodes that didn't appear in the constraints, otherwise SCIP will complain.
    used_nodes = set()
    for group in inequalities:
        used_nodes.update(group)
    for node in range(10):
        if node not in used_nodes:
            inequalities.add((node,))

    # Write problem to CPLEX LP file.
    with open(filepath, 'w') as lp_file:
        lp_file.write("maximize\nOBJ:" + "".join([f" + 1 x{node + 1}" for node in range(len(graph))]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, group in enumerate(inequalities):
            lp_file.write(f"C{count + 1}:" + "".join([f" + x{node + 1}" for node in sorted(group)]) + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node + 1}" for node in range(len(graph))]) + "\n")
