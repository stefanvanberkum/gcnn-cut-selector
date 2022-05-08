"""This is the main execution environment.

"""

from datetime import timedelta
from math import floor
from multiprocessing import cpu_count
from time import time

from numpy.random import default_rng

from instance_generator import generate_instances


def main():
    generate = True
    collect = True
    train = True
    test = True
    evaluate = True
    seed = 0

    n_jobs = cpu_count()
    seed_generator = default_rng(seed)
    seeds = seed_generator.integers(2 ** 32, size=1)
    start_time = time()

    if generate:
        generate_instances(seeds[0])
        print(f"Elapsed time: {str(timedelta(seconds=floor(time() - start_time)))}\n")


if __name__ == '__main__':
    main()
