"""
local_search.py – Intra/inter-machine local search moves.

[FIX] Para instancias grandes (≥100 ops) el coste del LS es prohibitivo:
  - local_search_intra evalúa O((n/M)^2 * M) pares por pass
  - _ls_until_convergence hace hasta 20 passes
  - local_search() con ils_iter=3 repite eso 4 veces
  En 6M140 esto equivale a ~188.000 evaluaciones por llamada, dejando
  solo 46 generaciones en 1000s y bloqueando al GA.

  Solución: _ls_until_convergence tiene un límite de passes adaptativo
  según n_ops (MAX_PASSES), y local_search() recibe ils_iter adaptativo
  desde ql_matheuristic (ya existía este parámetro, solo faltaba usarlo).

  Para instancias pequeñas (<60 ops) el comportamiento es idéntico al original.
"""
from __future__ import annotations
import random, itertools, time
from typing import List, Optional
from models import Instance, Chromosome
from evaluation import evaluate

LS_SWAP_SAMPLE = 60   # random pairs sampled per inter-swap pass


def _machine_positions(VM, machine):
    return [g for g, m in enumerate(VM) if m == machine]


def local_search_intra(chrom: Chromosome, instance: Instance) -> Chromosome:
    """Intra-machine resequencing – first-improvement, one pass per machine."""
    VI, VM = list(chrom[0]), list(chrom[1])
    current_fit = evaluate((VI, VM), instance)
    for m in instance.machines:
        positions = _machine_positions(VM, m)
        if len(positions) < 2: continue
        for idx_a, idx_b in itertools.combinations(range(len(positions)), 2):
            ga, gb = positions[idx_a], positions[idx_b]
            VI[ga], VI[gb] = VI[gb], VI[ga]
            new_fit = evaluate((VI, VM), instance)
            if new_fit < current_fit:
                current_fit = new_fit; break
            VI[ga], VI[gb] = VI[gb], VI[ga]
    return (VI, VM)


def local_search_inter_relocate(chrom: Chromosome, instance: Instance) -> Chromosome:
    """Inter-machine relocate – first-improvement, one pass."""
    VI, VM = list(chrom[0]), list(chrom[1])
    current_fit = evaluate((VI, VM), instance)
    if len(instance.machines) < 2: return (VI, VM)
    for g in range(len(VI)):
        orig_m = VM[g]
        for m in instance.machines:
            if m == orig_m: continue
            VM[g] = m; new_fit = evaluate((VI, VM), instance)
            if new_fit < current_fit:
                current_fit = new_fit; orig_m = m; break
            VM[g] = orig_m
    return (VI, VM)


def local_search_inter_swap(chrom: Chromosome, instance: Instance) -> Chromosome:
    """Inter-machine swap – first-improvement + random sampling."""
    VI, VM = list(chrom[0]), list(chrom[1])
    current_fit = evaluate((VI, VM), instance)
    n = len(VI)
    if len(instance.machines) < 2: return (VI, VM)
    all_pairs = [(ga, gb) for ga in range(n)
                           for gb in range(ga + 1, n) if VM[ga] != VM[gb]]
    pairs = (random.sample(all_pairs, LS_SWAP_SAMPLE)
             if LS_SWAP_SAMPLE and len(all_pairs) > LS_SWAP_SAMPLE
             else all_pairs)
    for ga, gb in pairs:
        if VM[ga] == VM[gb]: continue
        VI[ga], VI[gb] = VI[gb], VI[ga]; VM[ga], VM[gb] = VM[gb], VM[ga]
        new_fit = evaluate((VI, VM), instance)
        if new_fit < current_fit:
            current_fit = new_fit
        else:
            VI[ga], VI[gb] = VI[gb], VI[ga]; VM[ga], VM[gb] = VM[gb], VM[ga]
    return (VI, VM)


def _perturb(chrom: Chromosome, instance: Instance, k: int = 3) -> Chromosome:
    """Random kick: relocates k random jobs to random machines."""
    VI, VM = list(chrom[0]), list(chrom[1])
    machines = instance.machines
    if len(machines) < 2:
        return (VI, VM)
    for g in random.sample(range(len(VI)), min(k, len(VI))):
        others = [m for m in machines if m != VM[g]]
        VM[g] = random.choice(others)
    return (VI, VM)


def _max_passes(n_ops: int) -> int:
    """[FIX] Límite de passes de LS adaptativo al tamaño de instancia.

    El original usa 20 para todos los tamaños. Para instancias grandes
    eso es inasumible: cada pass cuesta ~2000-3000 evaluaciones en 6M140.

      n_ops <  60  →  20 passes  (igual que el original)
      n_ops <  100 →   5 passes
      n_ops >= 100 →   2 passes  (suficiente para pulir sin bloquear el GA)
    """
    if n_ops < 60:
        return 20
    elif n_ops < 100:
        return 5
    else:
        return 2


def _ls_until_convergence(chrom: Chromosome, instance: Instance) -> Chromosome:
    """Runs the full LS pipeline until no improvement."""
    max_p = _max_passes(instance.n_ops)   # [FIX] adaptativo
    prev_fit = evaluate(chrom, instance)
    for _ in range(max_p):
        chrom = local_search_intra(chrom, instance)
        chrom = local_search_inter_relocate(chrom, instance)
        chrom = local_search_inter_swap(chrom, instance)
        new_fit = evaluate(chrom, instance)
        if new_fit >= prev_fit - 1e-9:
            break
        prev_fit = new_fit
    return chrom


def local_search(chrom: Chromosome, instance: Instance, ils_iter: int = 3) -> Chromosome:
    """ILS: LS until convergence, perturb to escape local optima, repeat.

    ils_iter=0 → solo un paso de LS sin perturbaciones (más barato).
    El valor por defecto 3 se mantiene para compatibilidad con instancias pequeñas;
    ql_matheuristic pasa ils_iter=0 para instancias grandes.
    """
    best = _ls_until_convergence(chrom, instance)
    best_fit = evaluate(best, instance)
    for _ in range(ils_iter):
        candidate = _ls_until_convergence(_perturb(best, instance), instance)
        cand_fit = evaluate(candidate, instance)
        if cand_fit < best_fit - 1e-9:
            best, best_fit = candidate, cand_fit
    return best


# ── VM-only local search ──────────────────────────────────────────────────
# These variants keep VI (job order) fixed and only modify VM (machine assignments).
# Useful as a standalone baseline that isolates the machine-assignment dimension.

def local_search_vm_relocate(chrom: Chromosome, instance: Instance,
                              deadline: Optional[float] = None) -> Chromosome:
    """VM-only relocate: try reassigning each job to every other machine.

    VI is kept fixed. First-improvement, one pass over all jobs.
    Equivalent to local_search_inter_relocate but without touching VI.
    Stops early if deadline (absolute time.time() value) is reached.
    """
    VI, VM = list(chrom[0]), list(chrom[1])
    current_fit = evaluate((VI, VM), instance)
    if len(instance.machines) < 2:
        return (VI, VM)
    for g in range(len(VI)):
        if deadline and time.time() >= deadline:
            break
        orig_m = VM[g]
        for m in instance.machines:
            if m == orig_m:
                continue
            VM[g] = m
            new_fit = evaluate((VI, VM), instance)
            if new_fit < current_fit:
                current_fit = new_fit
                orig_m = m
                break
            VM[g] = orig_m
    return (VI, VM)


def local_search_vm_swap(chrom: Chromosome, instance: Instance,
                          deadline: Optional[float] = None) -> Chromosome:
    """VM-only swap: try swapping the machine assignments of two jobs on different machines.

    VI is kept fixed (job identities don't move, only their machine labels swap).
    First-improvement + random sampling.
    Stops early if deadline (absolute time.time() value) is reached.
    """
    VI, VM = list(chrom[0]), list(chrom[1])
    current_fit = evaluate((VI, VM), instance)
    n = len(VI)
    if len(instance.machines) < 2:
        return (VI, VM)
    all_pairs = [(ga, gb) for ga in range(n)
                           for gb in range(ga + 1, n) if VM[ga] != VM[gb]]
    pairs = (random.sample(all_pairs, LS_SWAP_SAMPLE)
             if LS_SWAP_SAMPLE and len(all_pairs) > LS_SWAP_SAMPLE
             else all_pairs)
    for ga, gb in pairs:
        if deadline and time.time() >= deadline:
            break
        if VM[ga] == VM[gb]:
            continue
        VM[ga], VM[gb] = VM[gb], VM[ga]
        new_fit = evaluate((VI, VM), instance)
        if new_fit < current_fit:
            current_fit = new_fit
        else:
            VM[ga], VM[gb] = VM[gb], VM[ga]
    return (VI, VM)


def _perturb_vm(chrom: Chromosome, instance: Instance, k: int = 3) -> Chromosome:
    """VM-only perturbation: randomly reassign k jobs to different machines.

    VI is kept fixed.
    """
    VI, VM = list(chrom[0]), list(chrom[1])
    machines = instance.machines
    if len(machines) < 2:
        return (VI, VM)
    for g in random.sample(range(len(VI)), min(k, len(VI))):
        others = [m for m in machines if m != VM[g]]
        VM[g] = random.choice(others)
    return (VI, VM)


def _ls_vm_until_convergence(chrom: Chromosome, instance: Instance,
                              deadline: Optional[float] = None) -> Chromosome:
    """VM-only LS pipeline: relocate → swap, repeated until no improvement or deadline."""
    max_p = _max_passes(instance.n_ops)
    prev_fit = evaluate(chrom, instance)
    for _ in range(max_p):
        if deadline and time.time() >= deadline:
            break
        chrom = local_search_vm_relocate(chrom, instance, deadline=deadline)
        chrom = local_search_vm_swap(chrom, instance, deadline=deadline)
        new_fit = evaluate(chrom, instance)
        if new_fit >= prev_fit - 1e-9:
            break
        prev_fit = new_fit
    return chrom


def local_search_vm(chrom: Chromosome, instance: Instance,
                    ils_iter: int = 3,
                    deadline: Optional[float] = None) -> Chromosome:
    """VM-only ILS: optimise machine assignments while keeping job order fixed.

    Runs _ls_vm_until_convergence, then perturbs VM and repeats ils_iter times,
    keeping the best solution found.

    ils_iter=0 → single LS pass with no perturbation (cheapest).
    deadline   → absolute time.time() value; search stops as soon as it is reached.
    """
    best = _ls_vm_until_convergence(chrom, instance, deadline=deadline)
    best_fit = evaluate(best, instance)
    for _ in range(ils_iter):
        if deadline and time.time() >= deadline:
            break
        candidate = _ls_vm_until_convergence(
            _perturb_vm(best, instance), instance, deadline=deadline)
        cand_fit = evaluate(candidate, instance)
        if cand_fit < best_fit - 1e-9:
            best, best_fit = candidate, cand_fit
    return best
