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
from .core.utils import FailedSubmissionError
from .core.job_environment import JobEnvironment as JobEnvironment
from .local.debug import DebugExecutor as DebugExecutor
from .local.debug import DebugJob as DebugJob
from .local.local import LocalExecutor as LocalExecutor
from .local.local import LocalJob as LocalJob
from .slurm.slurm import SlurmExecutor as SlurmExecutor
from .slurm.slurm import SlurmJob as SlurmJob
from .slurm.slurm import SlurmInfoWatcher as SlurmInfoWatcher

__version__ = "1.4.5"

import subprocess
import getpass
import datetime
import re

from functools import partial
import typing as tp
import sys
import os
import cloudpickle

from copy import deepcopy
from submitit import *
R = tp.TypeVar("R", covariant=True)

# We change the delay of the SlurmInfoWatcher to 1min s.t. we do not have to wait long for updates, after waiting long for scheduling
# the default is to increase the time until the next update of the state as the last submitted job is longer and longer ago up to `delay_s`
SlurmJob.watcher = SlurmInfoWatcher(delay_s=60)


def print_job_out(job, only_stdout=False, only_stderr=False, last_x_lines=None):
    assert not (only_stderr and only_stdout)

    any_print = False
    if not only_stderr and job.stdout() is not None:
        print("STD OUT")
        so = job.stdout().replace('\\n', '\n')
        print('\n'.join(so.split('\n')[-last_x_lines:]) if last_x_lines else so)
        any_print = True
    if not only_stdout and job.stderr() is not None:
        print("STD ERR")
        se = job.stderr().replace('\\n', '\n')
        print(se[-last_x_lines:] if last_x_lines else se)
        any_print = True
    if not any_print:
        print('No outputs yet. Probably because the job was not yet scheduled.')



Job.print = print_job_out
SlurmJob.print = print_job_out


def get_config(job):
    if hasattr(job, '_config'):
        return deepcopy(job._config)
    else:
        raise ValueError('Job has no config. You need to submit with `submit_group` for it to have one.')


config_getter = property(get_config)
Job.config = config_getter
SlurmJob.config = config_getter


def job_setstate(self, state):
    if 'config' in state:
        state['_config'] = state['config']
        del state['config']
    self.__dict__.update(state)

Job.__setstate__ = job_setstate
SlurmJob.__setstate__ = job_setstate


class JobGroup(list):
    def __init__(self, jobs, name):
        self.name = name
        super().__init__(jobs)
        
    def cancel(self):
        for job in self:
            job.cancel()

    def __repr__(self):
        return f"JobGroup[{self.name}]({super().__repr__()})"

class ConfigLoggingAutoExecutor(AutoExecutor):
    groups = {}

    def submit(self, fn: tp.Callable, *args: tp.Any, **kwargs: tp.Any) -> Job:
        """
        Submit the function `fn` to be executed on the cluster as `fn(*args, **kwargs)`.
        """
        try:
            return super().submit(fn, *args, **kwargs)
        except FailedSubmissionError as e:
            print(e)
            raise ValueError('We have a failed submission. This is likely due to selecting a non-existant partition or a too large time-out.')
            

    def submit_group(self, name: str, fn: tp.Callable, list_of_kwargs: tp.List[tp.Dict[str, tp.Any]], max_parallel=100):
        """

        :param name: Saves the job list in folder/name.joblist
        :param fn: The function, executed on each kwargs on the cluster
        :param list_of_kwargs:
        :return: job list
        """
        if "wandb" in sys.modules:
            print(
                "You might run into seriously weird errors if you use wandb in your notebook and your submitit job."
            )
            if input("Do you anyways want to continue? (y/n)") != "y":
                return
        job_list_fname = self.folder / (name + '.joblist')
        if name in self.groups or os.path.exists(job_list_fname):
            if input('Job list already exists. Overwrite? (y/n)') != 'y':
                return
        self.update_parameters(name=name, slurm_array_parallelism=max_parallel)
        fns = [partial(fn, **kwargs) for kwargs in list_of_kwargs]
        try:
            jobs = self.submit_array(fns)
        except FailedSubmissionError as e:
            print(e)
            raise ValueError('We have a failed submission. This is likely due to selecting a non-existant partition or a too large time-out.')
        for job, kwargs in zip(jobs, list_of_kwargs):
            job._config = kwargs
        self.groups[name] = jobs
        with open(job_list_fname, 'wb') as f:
            cloudpickle.dump(jobs, f)
        return JobGroup(jobs, name)



    def get_group(self, name: str) -> tp.List[Job]:
        if name not in self.groups:
            with open(self.folder / (name + '.joblist'), 'rb') as f:
                self.groups[name] = cloudpickle.load(f)
        return JobGroup(self.groups[name], name)

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


def get_executor(folder="~/submitit_logs", timeout_min=60, slurm_partition="testdlc_gpu-rtx2080",
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


def get_current_user() -> str:
    """
    Get the currently logged in unix user name.
    """
    return getpass.getuser()

def get_all_job_names_in_queue_by_user(username: str=get_current_user()) -> tp.List[str]:
    """
    This functions helps you to find out which jobs are in the queue, that you submitted.
    This includes both jobs which are running and pending.

    You can also specify another username, the default is your own username.
    """
    result = subprocess.run(['squeue', '--user', username, '--format', '%j'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    return output.split('\n')[1:-1]

def get_all_groups_used_for_jobs(job_names: tp.List[str]) -> tp.List[JobGroup]:
    """
    Get all the compute groups, used for these jobs or their descendents.
    """
    job_names = list(set(job_names))
    groups = {}
    for jn in job_names:
        try:
            groups[jn] = ex.get_group(jn)
        except FileNotFoundError:
            pass
    return groups


def get_all_groups_in_queue() -> tp.List[JobGroup]:
    """
    Get all groups that are currently queued by you.
    """
    return get_all_groups_used_for_job_names(get_all_job_names_in_queue_by_user())
