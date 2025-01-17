[![CircleCI](https://circleci.com/gh/facebookincubator/submitit.svg?style=svg)](https://circleci.com/gh/facebookincubator/workflows/submitit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pypi](https://img.shields.io/pypi/v/submitit)](https://pypi.org/project/submitit/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/submitit)](https://anaconda.org/conda-forge/submitit)
# Submit it!

## Install
Install via
  ```
  pip install git+https://github.com/automl/submitit@main#egg=submitit
  ```

If you have it installed, you can update it with the above comment, if the version number in `submitit/__init__.py` was updated, else you can use this

```
pip install git+https://github.com/automl/submitit@main#egg=submitit --force-reinstall --no-deps
```
for debugging.

## What is submitit?

Submitit is a lightweight tool for submitting Python functions for computation within a Slurm cluster.
It basically wraps submission and provide access to results, logs and more.
[Slurm](https://slurm.schedmd.com/quickstart.html) is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters.
Submitit allows to switch seamlessly between executing on Slurm or locally.

### An example is worth a thousand words: performing an addition

From inside an environment with `submitit` installed running on the login node (or from wherever you usually call `sbatch`):

```python
import submitit

def add(a, b):
    print('hi')
    return a + b

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.get_executor()
job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

output = job.result()  # waits for completion and returns output
assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster

#print job out/err
job.print()
```

**If you get `submitit.core.utils.FailedSubmissionError: Could not make sense of sbatch output`, see [below](#help).**

The `Job` class also provides tools for reading the log files (`job.stdout()` and `job.stderr()`).
`submitit.get_executor()` will by default request one GPU per job. Have a look at the arguments to learn more about it.
(They are very close to the usual SLURM arguments.)

If what you want to run is a command, turn it into a Python function using `submitit.helpers.CommandFunction`, then submit it.
By default stdout is silenced in `CommandFunction`, but it can be unsilenced with `verbose=True`.

### A group example

Our version of submitit supports groups.
Groups are named and are saved using their name on the cluster.
Thus, they persist on the cluster even when your notebook breaks down.

A group run is started like this:

```python
executor = submitit.get_executor()

def add(a, b):
    print('hi')
    return a + b

job_group = executor.submit_group('adding', add, [{'a': 4, 'b': 3}, {'a':1, 'b': 1}])

print(job_group)

for j in job_group:
    print(j.result())
    print('kwargs of this job', j.config)
    
# get your job group by its name, this even works after restarting your notebook or in a different notebook
job_group = executor.get_group('adding')
    
# job groups can also be canceled together
job_group.cancel()
```

**Find more examples [here](docs/examples.md)!!!**

Submitit is a Python 3.6+ toolbox for submitting jobs to Slurm.
It aims at running python function from python code.





## Help
There is one particularly bad error, that is
```
submitit.core.utils.FailedSubmissionError: Could not make sense of sbatch output ""
```

This means something went wrong in the communication between the cluster and `submitit`.
One can get more info by constructing the `sbatch` script, that `submitit` generates, by hand.
This takes some time, though. (We will add a tutorial to do this.)

So far this error meant one of the following (so playing with these might solve it quickly):
1. The partition is not supported for some reason, try another.
2. You specified a time out over the limit for that partition.
3. You specified `gpus_per_node=` on the executor (either in `get_executor` or in `update_parameters`)
4. Passing Arguments as the wrong type, i.e. the time out as `float` or `str` instead of `int`.

## Documentation

See the following pages for more detailled information:

- [Examples](docs/examples.md): for a bunch of examples dealing with errors, concurrency, multi-tasking etc...
- [Structure and main objects](docs/structure.md): to get a better understanding of how `submitit` works, which files are created for each job, and the main objects you will interact with.
- [Checkpointing](docs/checkpointing.md): to understand how you can configure your job to get checkpointed when preempted and/or timed-out.
- [Tips and caveats](docs/tips.md): for a bunch of information that can be handy when working with `submitit`.
- [Hyperparameter search with nevergrad](docs/nevergrad.md): basic example of `nevergrad` usage and how it interfaces with `submitit`.


### Goals

The aim of this Python3 package is to be able to launch jobs on Slurm painlessly from *inside Python*, using the same submission and job patterns than the standard library package `concurrent.futures`:

Here are a few benefits of using this lightweight package:
 - submit any function, even lambda and script-defined functions.
 - raises an error with stack trace if the job failed.
 - requeue preempted jobs (Slurm only)
 - swap between `submitit` executor and one of `concurrent.futures` executors in a line, so that it is easy to run your code either on slurm, or locally with multithreading for instance.
 - checkpoints stateful callables when preempted or timed-out and requeue from current state (advanced feature).
 - easy access to task local/global rank for multi-nodes/tasks jobs.
 - same code can work for different clusters thanks to a plugin system.

Submitit is used by FAIR researchers on the FAIR cluster.
The defaults are chosen to make their life easier, and might not be ideal for every cluster.

### Non-goals

- a commandline tool for running slurm jobs. Here, everything happens inside Python. To this end, you can however use [Hydra](https://hydra.cc/)'s [submitit plugin](https://hydra.cc/docs/next/plugins/submitit_launcher) (version >= 1.0.0).
- a task queue, this only implements the ability to launch tasks, but does not schedule them in any way.
- being used in Python2! This is a Python3.6+ only package :)


### Comparison with dask.distributed

[`dask`](https://distributed.dask.org/en/latest/) is a nice framework for distributed computing. `dask.distributed` provides the same `concurrent.futures` executor API as `submitit`:

```python
from distributed import Client
from dask_jobqueue import SLURMCluster
cluster = SLURMCluster(processes=1, cores=2, memory="2GB")
cluster.scale(2)  # this may take a few seconds to launch
executor = Client(cluster)
executor.submit(...)
```

The key difference with `submitit` is that `dask.distributed` distributes the jobs to a pool of workers (see the `cluster` variable above) while `submitit` jobs are directly jobs on the cluster. In that sense `submitit` is a lower level interface than `dask.distributed` and you get more direct control over your jobs, including individual `stdout` and `stderr`, and possibly checkpointing in case of preemption and timeout. On the other hand, you should avoid submitting multiple small tasks with `submitit`, which would create many independent jobs and possibly overload the cluster, while you can do it without any problem through `dask.distributed`.


## Contributors

By chronological order: Jérémy Rapin, Louis Martin, Lowik Chanussot, Lucas Hosseini, Fabio Petroni, Francisco Massa, Guillaume Wenzek, Thibaut Lavril, Vinayak Tantia, Andrea Vedaldi, Max Nickel, Quentin Duval (feel free to [contribute](.github/CONTRIBUTING.md) and add your name ;) )

## License

Submitit is released under the [MIT License](LICENSE).
