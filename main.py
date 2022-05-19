"""This is the main execution environment."""

from datetime import timedelta
from math import floor
from multiprocessing import cpu_count
from time import time

from numpy.random import default_rng

from data_collector import collect_data
from instance_generator import generate_instances
from model_evaluator import evaluate_models
from model_tester import test_models
from model_trainer import train_models
from utils import generate_seeds


def main():
    generate = True
    collect = False
    train = False
    test = False
    evaluate = False
    seed = 0

    n_jobs = cpu_count() - 6
    seed_generator = default_rng(seed)
    seeds = seed_generator.integers(2 ** 32, size=3)
    start_time = time()

    if generate:
        generate_instances(n_jobs, seeds[0])
        print(f"Elapsed time: {str(timedelta(seconds=floor(time() - start_time)))}\n")

    if collect:
        collect_data(n_jobs, seeds[1])
        print(f"Elapsed time: {str(timedelta(seconds=floor(time() - start_time)))}\n")

    if train:
        generate_seeds(seeds[2])
        train_models()
        print(f"Elapsed time: {str(timedelta(seconds=floor(time() - start_time)))}\n")

    if test:
        test_models()
        print(f"Elapsed time: {str(timedelta(seconds=floor(time() - start_time)))}\n")

    if evaluate:
        evaluate_models(n_jobs)
        print(f"Elapsed time: {str(timedelta(seconds=floor(time() - start_time)))}\n")


if __name__ == '__main__':
    main()
