"""This is the main execution environment."""

from datetime import timedelta
from math import ceil
from multiprocessing import cpu_count
from time import perf_counter

from data_collector import collect_data
from instance_generator import generate_instances
from model_evaluator import evaluate_models
from model_tester import test_models
from model_trainer import train_models
from summarizer import summarize_stats
from utils import generate_seeds


def main():
    """This method can be used to run everything.

    First, instances are generated from which we then sample expert (state, action) pairs. These observations are
    then used to train models, which are then tested and evaluated.

    All methods can also be run from the command line (see README).
    """

    # Run configurations.
    generate = True
    collect = True
    train = True
    test = True
    evaluate = False  # This needs to be run separately as TensorFlow cannot disable GPU mid-execution.
    summarize = False
    seed = 0

    n_jobs = cpu_count()

    generate_seeds(n_seeds=3, name='program_seeds', seed=seed)

    start_time = perf_counter()

    if generate:
        generate_instances(n_jobs)
        print(f"Elapsed time: {str(timedelta(seconds=ceil(perf_counter() - start_time)))}\n")

    if collect:
        collect_data(n_jobs)
        print(f"Elapsed time: {str(timedelta(seconds=ceil(perf_counter() - start_time)))}\n")

    if train:
        train_models()
        print(f"Elapsed time: {str(timedelta(seconds=ceil(perf_counter() - start_time)))}\n")

    if test:
        test_models()
        print(f"Elapsed time: {str(timedelta(seconds=ceil(perf_counter() - start_time)))}\n")

    if evaluate:
        evaluate_models(n_jobs)
        print(f"Elapsed time: {str(timedelta(seconds=ceil(perf_counter() - start_time)))}\n")

    if summarize:
        summarize_stats()
        print(f"Elapsed time: {str(timedelta(seconds=ceil(perf_counter() - start_time)))}\n")


if __name__ == '__main__':
    main()
