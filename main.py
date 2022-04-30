"""This is the main execution environment.

"""

from multiprocessing import cpu_count

from instance_generator import generate_instances


def main():
    generate = True
    collect = True
    train = True
    test = True
    evaluate = True

    n_jobs = cpu_count()

    if generate:
        generate_instances()


if __name__ == '__main__':
    main()
