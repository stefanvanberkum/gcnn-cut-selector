"""This module is used to collect data for training, validation, and testing.

Summary
=======
This module provides methods for collecting data for imitation learning, based on an expert decision rule that ranks
cuts by their bound improvement. The methods in this module are based on [1]_.

Classes
========
- :class:`SamplingAgent`: A cut selector used for sampling expert (state, action) pairs.

Functions
=========
- :func:`collect_data`: Collects data in accordance with our sampling scheme.
- :func:`collect_problem`: Collects samples for a single problem type.
- :func:`collect_samples`: Runs branch-and-cut episodes on the given set of instances, and collects randomly queried
  (state, action) pairs from an expert decision rule based on bound improvement.
- :func:`send_tasks`: Dispatcher loop: continuously send tasks to the task queue.
- :func:`generate_samples`: Worker loop: fetch an instance, run an episode, and send samples to the out queue.

References
==========
.. [1] Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). Exact combinatorial optimization with
    graph convolutional neural networks. *Neural Information Processing Systems (NeurIPS 2019)*, 15580–15592.
    https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html
"""

import glob
import gzip
import multiprocessing as mp
import os
import pickle
import shutil

import numpy as np
from pyscipopt import Model, SCIP_LPSOLSTAT, SCIP_RESULT
from pyscipopt.scip import Cutsel

import utils


class SamplingAgent(Cutsel):
    """Cut selector used for sampling expert (state, action) pairs.

    This class extends PySCIPOpt's Cutsel class for user-defined cut selector plugins.

    Methods
    =======
    - :meth:`cutselselect`: This method is called whenever cuts need to be ranked.

    :ivar episode: The episode number (instance/seed combination).
    :ivar instance: The filepath to the current instance.
    :ivar out_queue: The out queue where the sampling agent should send samples to.
    :ivar out_dir: The save file path for samples.
    :ivar seed: A seed value for the random number generator.
    :ivar p_expert: The probability of querying the expert on each cut selection round.
    :ivar p_max: The maximum parallelism for low-quality cuts.
    :ivar p_max_ub: The maximum parallelism for high-quality cuts.
    :ivar skip_factor: The factor that determines the high-quality threshold relative to the highest-quality cut.
    """

    def __init__(self, episode: int, instance: str, out_queue: mp.SimpleQueue, out_dir: str, seed: int, p_expert=0.05,
                 p_max=0.1, p_max_ub=0.5, skip_factor=0.9):
        self.episode = episode
        self.instance = instance
        self.out_queue = out_queue
        self.out_dir = out_dir
        self.seed = seed

        self.p_expert = p_expert
        self.p_max = p_max
        self.p_max_ub = p_max_ub
        self.skip_factor = skip_factor

        self.sample_counter = 0
        self.rng = np.random.default_rng(seed)

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):

        query_expert = self.rng.random() < self.p_expert
        if query_expert:
            # Rank all cuts based on their bound improvement, and record the expert (state, action) pair.
            quality = np.zeros(len(cuts))
            bound = self.model.getLPObjVal()

            # For each candidate cut, determine the relative bound improvement compared to the current LP solution.
            uneventful = True
            for i in range(len(cuts)):
                cut = cuts[i]
                self.model.startDive()

                # Add the cut to the LP relaxation and solve it.
                self.model.addRowDive(cut)
                self.model.constructLP()
                self.model.solveDiveLP()

                # Make sure that the LP bound is feasible.
                quality[i] = -np.Inf
                solstat = self.model.getLPSolstat()
                if solstat != SCIP_LPSOLSTAT.ITERLIMIT and solstat != SCIP_LPSOLSTAT.TIMELIMIT:
                    cut_bound = self.model.getLPObjVal()

                    # Compute the relative bound improvement.
                    quality[i] = abs(bound - cut_bound) / abs(bound)
                else:
                    # Do not record this (state, action) pair if SCIP hit a limit during solving.
                    uneventful = False
                    break
                self.model.endDive()

            if uneventful:
                # Record the state, action, and action set.
                state = utils.get_state(self.model, cuts)
                data = [state, quality, cuts]

                filename = f'{self.out_dir}/sample_{self.episode}_{self.sample_counter}.pkl'
                with gzip.open(filename, 'wb') as file:
                    pickle.dump({'data': data}, file)
                self.out_queue.put(
                    {'type': 'sample', 'episode': self.episode, 'instance': self.instance, 'filename': filename})
                self.sample_counter += 1

        if not query_expert or not uneventful:
            # Fall back to a hybrid branching rule.
            quality = [self.model.getCutEfficacy(cut) + 0.1 * self.model.getRowNumIntCols(
                cut) / cut.getNNonz() + 0.1 * self.model.getRowObjParallelism(
                cut) + 0.5 * self.model.getCutLPSolCutoffDistance(cut, self.model.getBestSol()) for cut in cuts]

        # Rank the cuts in descending order of quality.
        rankings = sorted(range(len(cuts)), key=lambda x: quality[x], reverse=True)
        sorted_cuts = np.array([cuts[rank] for rank in rankings])

        # Sort cut quality in descending order as well to match the array with sorted cuts.
        quality = -np.sort(-quality)

        # First check whether any cuts are parallel to forced cuts.
        n_selected = len(cuts)
        for cut in forcedcuts:
            # Mark all cuts that are parallel to forced cut i.
            parallelism = [self.model.getRowParallelism(cut, sorted_cuts[j]) for j in range(n_selected)]
            parallelism = np.pad(parallelism, (0, len(cuts) - n_selected), constant_values=0)
            marked = parallelism > self.p_max

            # Only remove low-quality or very parallel cuts.
            low_quality = np.logical_or(quality < 0.9 * quality[0], parallelism > self.p_max_ub)
            to_remove = np.logical_and(marked, low_quality)

            # Move cuts that are marked for removal to the back and decrease number of selected cuts.
            removed = sorted_cuts[to_remove]
            sorted_cuts = np.delete(sorted_cuts, to_remove)
            sorted_cuts = np.concatenate((sorted_cuts, removed))
            n_selected -= removed.size

        # Now remove cuts of low quality that are parallel to a cut of higher quality.
        i = 0
        while i < n_selected - 1:
            # Mark all cuts that are parallel to higher-quality cut i.
            parallelism = [self.model.getRowParallelism(sorted_cuts[i], sorted_cuts[j]) for j in
                           range(i + 1, len(sorted_cuts))]
            parallelism = np.pad(parallelism, (i + 1, 0), constant_values=0)
            marked = parallelism > self.p_max

            # Only remove low-quality or very parallel cuts.
            low_quality = np.logical_or(quality < 0.9 * quality[0], parallelism > self.p_max_ub)
            to_remove = np.logical_and(marked, low_quality)

            # Move cuts that are marked for removal to the back and decrease number of selected cuts.
            removed = sorted_cuts[to_remove]
            sorted_cuts = np.delete(sorted_cuts, to_remove)
            sorted_cuts = np.concatenate((sorted_cuts, removed))
            n_selected -= removed.size
            i += 1

        return {'cuts': sorted_cuts, 'nselectedcuts': min(n_selected, maxnselectedcuts), 'result': SCIP_RESULT.SUCCESS}


def collect_data(n_jobs: int, seed: int):
    """Collects samples for set covering, combinatorial auction, capacitated facility location, and independent set
    problems in accordance with our sampling scheme.

    :param n_jobs: The number of jobs to run in parallel.
    :param seed: A seed value for the random number generator.
    """

    seed_generator = np.random.default_rng(seed)
    seeds = seed_generator.integers(2 ** 32, size=4)

    print("Collecting set covering instance data...")
    train = glob.glob('data/instances/setcov/train_500r/*.lp')
    valid = glob.glob('data/instances/setcov/valid_500r/*.lp')
    test = glob.glob('data/instances/setcov/test_500r/*.lp')
    out_dir = 'data/samples/setcov/500r'
    collect_problem(train, valid, test, out_dir, n_jobs, seeds[0])
    print("")

    print("Collecting combinatorial auction instance data...")
    train = glob.glob('data/instances/combauc/train_100i_500b/*.lp')
    valid = glob.glob('data/instances/combauc/valid_100i_500b/*.lp')
    test = glob.glob('data/instances/combauc/test_100i_500b/*.lp')
    out_dir = 'data/samples/combauc/100i_500b'
    collect_problem(train, valid, test, out_dir, n_jobs, seeds[1])
    print("")

    print("Collecting capacitated facility instance data...")
    train = glob.glob('data/instances/capfac/train_100c/*.lp')
    valid = glob.glob('data/instances/capfac/valid_100c/*.lp')
    test = glob.glob('data/instances/capfac/test_100c/*.lp')
    out_dir = 'data/samples/capfac/100c'
    collect_problem(train, valid, test, out_dir, n_jobs, seeds[2])
    print("")

    print("Collecting maximum independent set instance data...")
    train = glob.glob('data/instances/indset/train_500n/*.lp')
    valid = glob.glob('data/instances/indset/valid_500n/*.lp')
    test = glob.glob('data/instances/indset/test_500n/*.lp')
    out_dir = 'data/samples/indset/500n'
    collect_problem(train, valid, test, out_dir, n_jobs, seeds[3])


def collect_problem(train: list[str], valid: list[str], test: list[str], out_dir: str, n_jobs: int, seed: int,
                    n_train=100000, n_valid=20000, n_test=20000):
    """Collects samples for a single problem type.

    :param train: A list of filepaths to training instances.
    :param valid: A list of filepaths to validation instances.
    :param test: A list of filepaths to testing instances.
    :param out_dir: The desired location for saving the collected samples.
    :param n_jobs: The number of jobs to run in parallel.
    :param seed: A seed value for the random number generator.
    :param n_train: The desired number of training samples.
    :param n_valid: The desired number of validation samples.
    :param n_test: The desired number of testing samples.
    """

    os.makedirs(out_dir)
    seed_generator = np.random.default_rng(seed)
    seeds = seed_generator.integers(2 ** 32, size=3)

    rng = np.random.default_rng(seeds[0])
    collect_samples(train, out_dir + '/train', rng, n_train, n_jobs)

    rng = np.random.default_rng(seeds[1])
    collect_samples(valid, out_dir + '/valid', rng, n_valid, n_jobs)

    rng = np.random.default_rng(seeds[2])
    collect_samples(test, out_dir + '/test', rng, n_test, n_jobs)


def collect_samples(instances: list[str], out_dir: str, rng: np.random.Generator, n_samples: int, n_jobs: int):
    """Runs branch-and-cut episodes on the given set of instances, and collects randomly queried (state,
    action) pairs from an expert decision rule based on bound improvement.

    The sampling is parallelized over multiple cores. A dispatcher sends tasks to a task queue, which are then
    processed by the workers. The workers send the results to an out queue, and finished samples are written to a
    file. The task queue contains 'start' orders, which signal a worker to start working on a given instance with a
    given seed value (an episode). The out queue contains finished samples (orders of type 'sample'), 'start' orders
    that signal that an episode has started, and 'done' orders that signal that an episode is finished. Multiple
    episodes are processed concurrently, but samples are written one episode at a time.

    :param instances: A list of filepaths to instances to use for sampling.
    :param n_samples: The desired number of samples.
    :param out_dir: The desired location for saving the collected samples.
    :param n_jobs: The number of jobs to run in parallel.
    :param rng: A random number generator.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Start workers, which process orders from the in queue and send samples to the out queue.
    task_queue = mp.Queue(maxsize=2 * n_jobs)
    out_queue = mp.SimpleQueue()
    workers = []
    for i in range(n_jobs):
        p = mp.Process(target=generate_samples, args=(task_queue, out_queue), daemon=True)
        workers.append(p)
        p.start()

    # Create a temporary directory.
    tmp_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    # Start dispatcher, which sends tasks to the in queue.
    dispatcher = mp.Process(target=send_tasks, args=(task_queue, instances, rng.integers(2 ** 32), tmp_dir),
                            daemon=True)
    dispatcher.start()

    # Record and write finished samples received in the out queue.
    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    progress = 1
    while i < n_samples:
        sample = out_queue.get()

        # Add received sample to buffer.
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # Write samples from current episode, if any.
        while current_episode in buffer and buffer[current_episode]:
            samples = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples:
                if sample['type'] == 'done':
                    # Received signal that the optimization for this episode finished, so move on to next episode.
                    del buffer[current_episode]
                    current_episode += 1
                else:
                    # Write sample.
                    os.rename(sample['filename'], f'{out_dir}/sample_{i + 1}.pkl')
                    in_buffer -= 1
                    i += 1

                    # Stop the dispatcher as soon as the number of samples collected and in buffer exceeds the
                    # required number of samples.
                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher.terminate()

                    # As soon as the required number of samples is collected, clear the buffer and stop.
                    if i == n_samples:
                        buffer = {}
                        break
        if i / n_samples > 0.1 * progress:
            print(f"Progress: {progress}0%")
            progress += 1

    # Stop all workers.
    for p in workers:
        p.terminate()

    # Remove temporary directory.
    shutil.rmtree(tmp_dir, ignore_errors=True)


def send_tasks(task_queue, instances, out_dir, seed):
    """Dispatcher loop: continuously send tasks to the task queue.

    The tasks signal a worker to start working on a given instance with a given seed value (an episode). This loop
    relies on limited queue capacity, as it continuously sends tasks.

    :param task_queue: The task queue to send tasks to.
    :param instances: The list of possible instances to sample from.
    :param out_dir: The desired location for saving the collected samples.
    :param seed: A seed value for the random number generator.
    """

    rng = np.random.default_rng(seed)

    episode = 0
    while True:
        instance = rng.choice(instances)
        seed = rng.integers(2 ** 32)
        task_queue.put([episode, instance, out_dir, seed])
        episode += 1


def generate_samples(task_queue, out_queue):
    """Worker loop: fetch an instance, run an episode, and send samples to the out queue.

    :param task_queue: The task queue from which the worker needs to fetch tasks.
    :param out_queue: The out queue to which the worker should send finished samples and 'done' orders.
    """

    while True:
        # Fetch a task.
        episode, instance, out_dir, seed = task_queue.get()

        # Initialize the model.
        model = Model()
        utils.init_scip(model, seed)
        model.readProblem(f'{instance}')

        # Include the sampling agent as cut selector to extract (state, action) pairs.
        cut_selector = SamplingAgent(episode, instance, out_queue, out_dir, seed)
        model.includeCutsel(cut_selector, 'sampler', 'samples expert decisions', 5000000)

        # Signal that the episode has started.
        out_queue.put({'type': 'start', 'episode': episode})

        # Process the episode.
        model.optimize()
        model.freeProb()

        # Signal that the episode has finished.
        out_queue.put({'type': 'done', 'episode': episode})
