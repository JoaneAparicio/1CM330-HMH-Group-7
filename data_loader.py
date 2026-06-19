"""
data_loader.py – CSV loader and experiment runners.

Loads instances from .csv files, runs the PH, MH, MH+LS algorithms multiple times to collect statistics, 
and provides a function to run a full comparison on a given instance. 
ARPD is computed against the PH objective value as reference.
"""
from __future__ import annotations
import csv
import random
from typing import Dict, Optional
from dataclasses import dataclass

from models import Instance, Operation, Chromosome
from evaluation import evaluate
from ga_operators import GAParams, practitioner_heuristic, matheuristic_ls, matheuristic_parallel_vm_ls

@dataclass
class GAParams(GAParams):
    pass   # re-export for convenience


def load_instance_csv(filepath: str, tau: float = 1.0,
                      wd: float = 1.0, ws: float = 1.0) -> Instance:
    """Parse a .csv file from vinhise/pmstr-basecases."""
    with open(filepath, newline="") as f:
        rows = list(csv.reader(f))
    meta = {}; header_end = 0
    for i, row in enumerate(rows):
        if row and row[0].strip() in ("O", "M", "T", "C"):
            meta[row[0].strip()] = int(row[1].strip()); header_end = i
        elif row and row[0].strip() == "Job":
            header_end = i; break
    capacity   = meta.get("C", 80)
    n_machines = meta.get("M", 2)
    operations = []; tool_sizes = {}
    for row in rows[header_end+1:]:
        if not row or not row[0].strip(): continue
        try:
            job_id = int(row[0]); op_idx = int(row[1])
            r = float(row[2]); p = float(row[3]); d = float(row[4])
            tool = int(row[5]); sz = int(row[6])
        except (ValueError, IndexError):
            continue
        tool_sizes[tool] = sz
        operations.append(Operation(job_id=job_id, op_idx=op_idx,
                                    release_time=r, proc_time=p,
                                    due_date=d, tool_set=tool))
    from models import Instance
    return Instance(operations=operations, machines=list(range(n_machines)),
                    tool_sizes=tool_sizes, magazine_capacity=capacity,
                    tool_setup_time=tau, wd=wd, ws=ws)


def run_experiment(instance, params, n_runs=10, best_known=None):
    """Run the base MH n_runs times and collect statistics. ARPD is computed against best_known when provided (should be ph_obj),
    otherwise falls back to the best value found across runs.

    instance: the problem instance to solve
    params: GAParams object with parameters for the MH
    n_runs: how many times to run the MH (with different seeds)
    best_known: optional reference value for ARPD calculation (e.g. PH objective)
    """
    import copy, time
    from ga_operators import matheuristic
    results = []
    for run in range(n_runs):
        p = GAParams(**{**params.__dict__, "seed": params.seed + run})
        r = matheuristic(instance, p); results.append(r)
        print(f"  Run {run+1:2d}: fitness={r.best_fitness:.4f}  "
              f"time={r.computation_time:.1f}s  gens={r.n_generations}")
    fitnesses = [r.best_fitness for r in results]
    times     = [r.computation_time for r in results]
    f_best    = min(fitnesses)
    # [FIX-6] use the external reference (ph_obj) when available
    ref  = best_known if best_known is not None else f_best
    arpd = 100.0 * sum((f - ref) / max(ref, 1e-9) for f in fitnesses) / n_runs if ref > 0 else 0.0
    mean = sum(fitnesses) / n_runs
    return {
        "mean_fitness": mean,
        "std_fitness":  (sum((f - mean)**2 for f in fitnesses) / n_runs) ** 0.5,
        "best_fitness": f_best,
        "mean_time":    sum(times) / n_runs,
        "arpd":         arpd,
        "all_results":  results,
    }


def run_experiment_ls(instance, params, n_runs=10, best_known=None):
    """Run the MH+LS variant n_runs times and collect statistics.

    Identical to run_experiment but calls matheuristic_ls() instead of
    matheuristic(), applying a Local Search post-processing step after the GA.

    instance: the problem instance to solve
    params: GAParams object with parameters for the MH
    n_runs: how many times to run (with different seeds)
    best_known: optional reference value for ARPD calculation (e.g. PH objective)
    """
    import copy, time
    from ga_operators import matheuristic_ls
    results = []
    for run in range(n_runs):
        p = GAParams(**{**params.__dict__, "seed": params.seed + run})
        r = matheuristic_ls(instance, p); results.append(r)
        print(f"  Run {run+1:2d}: fitness={r.best_fitness:.4f}  "
              f"time={r.computation_time:.1f}s  gens={r.n_generations}")
    fitnesses = [r.best_fitness for r in results]
    times     = [r.computation_time for r in results]
    f_best    = min(fitnesses)
    ref  = best_known if best_known is not None else f_best
    arpd = 100.0 * sum((f - ref) / max(ref, 1e-9) for f in fitnesses) / n_runs if ref > 0 else 0.0
    mean = sum(fitnesses) / n_runs
    return {
        "mean_fitness": mean,
        "std_fitness":  (sum((f - mean)**2 for f in fitnesses) / n_runs) ** 0.5,
        "best_fitness": f_best,
        "mean_time":    sum(times) / n_runs,
        "arpd":         arpd,
        "all_results":  results,
    }


def run_experiment_parallel_vm_ls(instance, params, n_runs=10, best_known=None):
    """Run the MH+PLS (parallel VM-LS) variant n_runs times and collect statistics.

    Calls matheuristic_parallel_vm_ls() which runs crossover, mutation and VM-only
    local search in parallel for each individual every generation, then does a
    4-way selection to keep the best.  Budget is controlled by eval function calls
    (params.max_evals).

    instance   : the problem instance to solve
    params     : GAParams with max_evals set (default 10_000 if absent)
    n_runs     : how many times to run (with different seeds)
    best_known : optional reference value for ARPD (e.g. ph_obj)
    """
    results = []
    for run in range(n_runs):
        p = GAParams(**{**params.__dict__, "seed": params.seed + run})
        r = matheuristic_parallel_vm_ls(instance, p)
        results.append(r)
        evals_used = getattr(p, "max_evals", 10_000)
        print(f"  Run {run+1:2d}: fitness={r.best_fitness:.4f}  "
              f"time={r.computation_time:.1f}s  gens={r.n_generations}  "
              f"max_evals={evals_used}")
    fitnesses = [r.best_fitness for r in results]
    times     = [r.computation_time for r in results]
    f_best    = min(fitnesses)
    ref  = best_known if best_known is not None else f_best
    arpd = 100.0 * sum((f - ref) / max(ref, 1e-9) for f in fitnesses) / n_runs if ref > 0 else 0.0
    mean = sum(fitnesses) / n_runs
    return {
        "mean_fitness": mean,
        "std_fitness":  (sum((f - mean)**2 for f in fitnesses) / n_runs) ** 0.5,
        "best_fitness": f_best,
        "mean_time":    sum(times) / n_runs,
        "arpd":         arpd,
        "all_results":  results,
    }

def run_comparison_collect(csv_path, label, max_time=120.0,
                           np_size=100, Gc=50, n_runs=10):
    """Run PH, MH and MH+LS on the given instance and collect results in a dict.

    PH is run n_runs times to get an average objective value used as ARPD reference.

    csv_path: path to the instance .csv file
    label: a descriptive label for the instance (used in printouts and results dict)
    max_time: maximum time in seconds for each run
    np_size: population size
    Gc: number of generations with no improvement to stop
    n_runs: how many times to run each algorithm (with different seeds)
    """
    print(f"\n{'='*60}\nInstance: {label}\n{'='*60}")
    inst = load_instance_csv(csv_path)
    print(f"  {inst.n_ops} ops | {inst.n_machines} machines | "
          f"{len(inst.tool_sizes)} tool sets | capacity={inst.magazine_capacity}")

    params = GAParams(np_size=np_size, max_time=max_time, Gc=Gc, seed=42)

    ph_results = []
    for run in range(n_runs):
        random.seed(run)
        _, ph_val = practitioner_heuristic(inst)
        ph_results.append(ph_val)
    ph_obj = sum(ph_results) / len(ph_results)
    ph_std = (sum((v - ph_obj)**2 for v in ph_results) / len(ph_results)) ** 0.5
    print(f"  PH objective : {ph_obj:.2f}±{ph_std:.2f}\n")

    print(f"  [MH]  running {n_runs} runs …")
    exp_mh = run_experiment(inst, params, n_runs=n_runs, best_known=ph_obj)
    mh_details = [(r.best_fitness, r.computation_time, r.n_generations)
                  for r in exp_mh["all_results"]]
    print(f"  MH    mean={exp_mh['mean_fitness']:.4f} ± {exp_mh['std_fitness']:.4f}  "
          f"best={exp_mh['best_fitness']:.4f}  time={exp_mh['mean_time']:.1f}s  "
          f"ARPD={exp_mh['arpd']:.2f}%")

    print(f"  [MH+LS] running {n_runs} runs …")
    exp_mhls = run_experiment_ls(inst, params, n_runs=n_runs, best_known=ph_obj)
    mhls_details = [(r.best_fitness, r.computation_time, r.n_generations)
                    for r in exp_mhls["all_results"]]
    print(f"  MH+LS mean={exp_mhls['mean_fitness']:.4f} ± {exp_mhls['std_fitness']:.4f}  "
          f"best={exp_mhls['best_fitness']:.4f}  time={exp_mhls['mean_time']:.1f}s  "
          f"ARPD={exp_mhls['arpd']:.2f}%")

    imp_mhls_vs_mh = 100 * (exp_mh["mean_fitness"] - exp_mhls["mean_fitness"]) / max(exp_mh["mean_fitness"], 1e-9)
    imp_mhls_vs_ph = 100 * (ph_obj - exp_mhls["best_fitness"]) / max(ph_obj, 1e-9)
    print(f"  MH→MH+LS: {imp_mhls_vs_mh:+.2f}%   PH→MH+LS: {imp_mhls_vs_ph:+.1f}%")

    return {
        "label":      label,
        "n_ops":      inst.n_ops,
        "n_machines": inst.n_machines,
        "n_tools":    len(inst.tool_sizes),
        "capacity":   inst.magazine_capacity,
        "ph_obj":     ph_obj,
        "mh_mean":    exp_mh["mean_fitness"],    "mh_std":    exp_mh["std_fitness"],
        "mh_best":    exp_mh["best_fitness"],    "mh_time":   exp_mh["mean_time"],
        "mh_arpd":    exp_mh["arpd"],
        "mhls_mean":  exp_mhls["mean_fitness"],  "mhls_std":  exp_mhls["std_fitness"],
        "mhls_best":  exp_mhls["best_fitness"],  "mhls_time": exp_mhls["mean_time"],
        "mhls_arpd":  exp_mhls["arpd"],
        "imp_mhls_vs_mh": imp_mhls_vs_mh,
        "imp_mhls_vs_ph": imp_mhls_vs_ph,
        "mh_run_details":   mh_details,
        "mhls_run_details": mhls_details,
    }
