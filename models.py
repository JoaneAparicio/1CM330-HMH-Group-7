"""
models.py – Data structures and chromosome encoding.

This module defines the core data structures for representing the scheduling problem instance,
including operations, jobs, machines, and the chromosome encoding used in the genetic algorithm. 

It also includes utility functions for decoding chromosomes and generating random chromosomes, 
as well as a function to build the example instance from the paper.
"""
# ── Imports ──────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    print("[WARNING] PuLP not found – TRM will use greedy fallback instead of ILP.")

# ── Operations ──────────────────────────────────────────────────────────────
@dataclass
class Operation:
    """Single operation (i, j) of job i."""
    job_id: int
    op_idx: int  # operation index within the job
    release_time: float
    proc_time: float
    due_date: float
    tool_set: int

# ── Problem Instance ────────────────────────────────────────────────────────
@dataclass
class Instance:
    """Full problem instance."""
    operations: List[Operation]
    machines: List[int]
    tool_sizes: Dict[int, int] # lot size, the magazine capacity, and tool_setup_time τ.
    magazine_capacity: int
    tool_setup_time: float
    wd: float = 1.0         
    ws: float = 1.0            
    n_jobs: int = field(init=False)
    jobs: Dict[int, List[Operation]] = field(init=False)

    def __post_init__(self):
        """Groups operations by job and sorts them by index."""
        self.jobs = {}
        for op in self.operations:
            self.jobs.setdefault(op.job_id, []).append(op)
        for ops in self.jobs.values():
            ops.sort(key=lambda o: o.op_idx)
        self.n_jobs = len(self.jobs)

    @property
    def n_machines(self) -> int:
        """Total number of machines."""
        return len(self.machines)

    @property
    def n_ops(self) -> int:
        """Total number of operations."""
        return len(self.operations)


# Chromosome = (VI, VM)
# VI[g] = job_id  -> job-order vector (the "sequence gene")
# VM[g] = machine_id  -> machine assignment vector (the "machine gene")
Chromosome = Tuple[List[int], List[int]]


def decode_sequence(VI: List[int], instance: Instance) -> List[Operation]:
    """Decodes the job-order vector into a sequence of operations."""
    counters: Dict[int, int] = {jid: 0 for jid in instance.jobs}
    seq: List[Operation] = []
    for job_id in VI:
        k = counters[job_id]
        seq.append(instance.jobs[job_id][k])
        counters[job_id] += 1
    return seq


def random_VI(instance: Instance) -> List[int]:
    """Generates a random valid job-order vector."""
    VI: List[int] = []
    for job_id, ops in instance.jobs.items():
        VI.extend([job_id] * len(ops))
    random.shuffle(VI)
    return VI


def random_VM(instance: Instance) -> List[int]:
    """Generates a random valid machine-assignment vector."""
    import random
    return [random.choice(instance.machines) for _ in range(instance.n_ops)]


def init_random_chromosome(instance: Instance) -> Chromosome:
    """Generates a random valid chromosome."""
    return (random_VI(instance), random_VM(instance))
