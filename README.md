# GCNN cut selector

A graph convolutional neural network (GCNN) for cutting plane selection.

## Installation

- If you do not have Conda, you can download Anaconda or Miniconda following the instructions listed
  here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html (we used Anaconda).
- Create a virtual environment with all required packages by running the following
  command: ```conda env create -f environment.yml```
- For benchmarking, download the MIPLIB 2010 benchmark set from: http://miplib2010.zib.de/.
    - Alternatively, one can download any other benchmarking set (e.g., MIPLIB 2017), as long as it contains instances
      that
      SCIP can read. For a list of file types, see: https://www.scipopt.org/doc/html/group__FILEREADERS.php.
    - Save the instances under ```path_to_code/data/benchmarking```, where ```path_to_code``` denotes the path to the
      project's directory.
- You can open and edit the code in any editor, we used the PyCharm IDE: https://www.jetbrains.com/pycharm/.

## Usage

The experiments can either be run from an editor (one long run) or via the
command line (many short runs). If you run on a single machine, running from the editor is the most straightforward. The
command line is especially useful if you would like to run multiple jobs concurrently. Both the editor and command line
make use of
parallelization wherever possible.

### Editor

The ```main.py``` module can be used to run all the experiments, simply by specifying what should be run in the run
configurations.

_Note._ When using GPU for TensorFlow, if any memory errors occur during model training, restarting your computer might
help.

### Command line

The following commands can be used to run all the experiments.

#### Instance generation

```
python instance_generator.py
```

Optional arguments:

- --n_jobs: The number of jobs to run in parallel (default: all cores).
- --seed: The seed value used for the program.

#### Sampling

The following loop runs all sampling sequentially, but it can be split up into parts to run concurrently.

```
for problem in setcov combauc capfac indset
do
  for set in train valid test
  do
    python data_collector.py $problem $set
  done
done
```

Optional argument:

- --n_jobs: The number of jobs to run in parallel (default: all cores).

#### Model training

The following loop runs all training sequentially, but it can be split up into parts to run concurrently.

```
for problem in setcov combauc capfac indset
do
  for iteration in {1..5}
  do
    python model_trainer.py $problem $iteration
  done
done
```

#### Model testing

The following loop runs all testing sequentially, but it can be split up into parts to run concurrently.

```
for problem in setcov combauc capfac indset
do
  for iteration in {1..5}
  do
    python model_tester.py $problem $iteration
  done
done
```

#### Model evaluation

```
python model_evaluator.py
```

Optional argument:

- --n_jobs: The number of jobs to run in parallel (default: all cores).

#### Model benchmarking

```
python model_benchmarker.py
```

Optional argument:

- --n_jobs: The number of jobs to run in parallel (default: all cores).

#### Summarizing

```
python summarizer.py
```

## File description

- ```data_collector.py```
    - Module used for sampling expert (state, action) pairs.
    - Parallelized over multiple cores.
- ```instance_generator.py```
    - Module used for randomly generating set covering, combinatorial auction, capacitated facility location, and
      maximum independent set problem instances.
- ```main.py```
    - Main execution environment.
- ```model.py```
    - Module that provides the GCNN model functionality.
- ```model_benchmarker.py```
    - Module used for benchmarking the GCNN approach.
    - Parallelized over multiple cores.
- ```model_evaluator.py```
    - Module used for evaluating trained models.
    - Parallelized over multiple cores.
- ```model_tester.py```
    - Module used for testing trained models.
- ```model_trainer.py```
    - Module used for training models.
- ```summarizer.py```
    - Module used to summarize all obtained results.
- ```utils.py```
    - Module that provides some general utility methods.

## Timings
