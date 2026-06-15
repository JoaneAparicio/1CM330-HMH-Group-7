"""
models.py – Data structures and chromosome encoding.

This module defines the core data structures for representing the scheduling problem instance,
including operations, jobs, machines, and the chromosome encoding used in the genetic algorithm. 

It also includes utility functions for decoding chromosomes and generating random chromosomes, 
as well as a function to build the example instance from the paper.
"""
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


@dataclass
class Operation:
    """Single operation (i, j) of job i."""
    job_id: int
    op_idx: int  # operation index within the job
    release_time: float
    proc_time: float
    due_date: float
    tool_set: int


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


def build_example1() -> Instance:
    """Table 1 from the paper: 8 jobs, 2 machines, 7 tool sets."""
    tool_sizes = {1: 21, 2: 12, 3: 19, 4: 19, 5: 52, 6: 53, 7: 55}
    raw = [
        (1,1,0,6,9,5),(2,1,4,8,13,4),(3,1,6,5,14,1),(4,1,6,7,17,4),
        (5,1,10,3,15,2),(1,2,28,5,33,1),(6,1,25,6,35,7),(8,1,33,6,39,6),
        (7,1,12,4,16,3),(3,2,20,5,25,2),
    ]
    operations = [
        Operation(job_id=r[0], op_idx=r[1], release_time=r[2],
                  proc_time=r[3], due_date=r[4], tool_set=r[5])
        for r in raw
    ]
    return Instance(operations=operations, machines=[0,1],
                    tool_sizes=tool_sizes, magazine_capacity=80,
                    tool_setup_time=1.0)
