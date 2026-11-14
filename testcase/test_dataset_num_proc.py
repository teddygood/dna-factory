"""
Test cases for dna_factory/training_runner.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory import training_runner


def test_user_dataset_num_proc_is_kept(monkeypatch):
    monkeypatch.setattr(training_runner.os, "process_cpu_count", lambda: 224)
    training_args = SimpleNamespace(dataset_num_proc=8)
    training_runner.fill_dataset_num_proc(training_args)
    assert training_args.dataset_num_proc == 8


def test_unset_dataset_num_proc_is_filled_with_half_the_cpus(monkeypatch):
    monkeypatch.setattr(training_runner.os, "process_cpu_count", lambda: 224)
    training_args = SimpleNamespace(dataset_num_proc=None)
    training_runner.fill_dataset_num_proc(training_args)
    assert training_args.dataset_num_proc == 112


def test_filled_dataset_num_proc_is_at_least_one(monkeypatch):
    for cpus in (1, None):
        monkeypatch.setattr(training_runner.os, "process_cpu_count", lambda c=cpus: c)
        training_args = SimpleNamespace(dataset_num_proc=None)
        training_runner.fill_dataset_num_proc(training_args)
        assert training_args.dataset_num_proc == 1
