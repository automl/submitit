# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""""Python 3.6+ toolbox for submitting jobs to Slurm"""

# allow explicit reimports (mypy) by renaming all imports
from . import helpers as helpers
from .auto.auto import AutoExecutor as AutoExecutor
from .core.core import Executor as Executor
from .core.core import Job as Job
from .core.job_environment import JobEnvironment as JobEnvironment
from .local.debug import DebugExecutor as DebugExecutor
from .local.debug import DebugJob as DebugJob
from .local.local import LocalExecutor as LocalExecutor
from .local.local import LocalJob as LocalJob
from .slurm.slurm import SlurmExecutor as SlurmExecutor
from .slurm.slurm import SlurmJob as SlurmJob

__version__ = "1.4.5"


from functools import partial
import typing as tp
import os
import cloudpickle

from submitit import *
R = tp.TypeVar("R", covariant=True)


def print_job_out(job, only_stdout=False, only_stderr=False, last_x_lines=None):
    assert not (only_stderr and only_stdout)
    if not only_stderr:
        print("STD OUT")
        so = job.stdout().replace('\\n', '\n')
        print('\n'.join(so.split('\n')[-last_x_lines:]) if last_x_lines else so)
    if not only_stdout:
        print("STD ERR")
        se = job.stderr().replace('\\n', '\n')
        print(se[-last_x_lines:] if last_x_lines else se)


Job.print = print_job_out
SlurmJob.print = print_job_out


class JobGroup(list):
    def cancel(self):
        for job in self:
            job.cancel()

    def __repr__(self):
        return f"JobGroup({super().__repr__()})"

class ConfigLoggingAutoExecutor(AutoExecutor):
    groups = {}

    def submit_group(self, name: str, fn: tp.Callable, list_of_kwargs: tp.List[tp.Dict[str, tp.Any]], max_parallel=100):
        """

        :param name: Saves the job list in folder/name.joblist
        :param fn: The function, executed on each kwargs on the cluster
        :param list_of_kwargs:
        :return: job list
        """
        job_list_fname = self.folder / (name + '.joblist')
        if name in self.groups or os.path.exists(job_list_fname):
            assert input('Job list already exists. Overwrite? (y/n)') == 'y'
        self.update_parameters(name=name, slurm_array_parallelism=max_parallel)
        fns = [partial(fn, **kwargs) for kwargs in list_of_kwargs]
        jobs = super().submit_array(fns)
        for job, kwargs in zip(jobs, list_of_kwargs):
            job.config = kwargs
        self.groups[name] = jobs
        with open(job_list_fname, 'wb') as f:
            cloudpickle.dump(jobs, f)
        return JobGroup(jobs)

    def get_group(self, name: str) -> tp.List[Job]:
        if name not in self.groups:
            with open(self.folder / (name + '.joblist'), 'rb') as f:
                self.groups[name] = cloudpickle.load(f)
        return JobGroup(self.groups[name])

    def is_group(self, name):
        try:
            self.get_group(name)
            return True
        except FileNotFoundError:
            return False

    def list_groups(self):
        return list(self.groups.keys())

    def print_job(self, group, index, **kwargs):
        print_job_out(self.get_group(group)[index], **kwargs)


def get_executor(folder="/work/dlclarge1/muellesa-MySpace", timeout_min=60, slurm_partition="testdlc_gpu-rtx2080",
                 slurm_gres='gpu:1', slurm_setup=['export MKL_THREADING_LAYER=GNU'], **kwargs):
    # executor is the submission interface (logs are dumped in the folderj)
    executor = ConfigLoggingAutoExecutor(folder=folder)
    # set timeout in min, and partition for running the job
    executor.update_parameters(timeout_min=timeout_min, slurm_partition=slurm_partition, slurm_gres=slurm_gres,
                               slurm_setup=slurm_setup, **kwargs)
    return executor


def print_job_states(jobs):
    for j in jobs:
        print(j.state)
