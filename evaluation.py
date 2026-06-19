"""
evaluation.py – Tool Replacement Method (TRM) and fitness evaluation.

This module computes how good a chromosome is by simulating the schedule and applying the TRM to decide which tools to evict when a magazine is full. 
It includes both a basic evaluation function that returns the total fitness and a more detailed one that also breaks down the fitness contributions by machine.
"""
# ── Imports ──────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import List, Dict, Tuple
from models import Instance, Chromosome, decode_sequence, HAS_PULP

try:
    import pulp
except ImportError:
    pass


def _trm_ilp(tm, ts_needed, phi_s, instance):
    """Computes the TRM eviction set using ILP. Falls back to a greedy heuristic if PuLP is not available.
    
    tm: current magazine (tool -> size)
    ts_needed: tool -> score (how urgently we need this tool in the future)
    phi_s: how much space we need to free
    instance: the problem instance (for tool sizes and capacity)
    """
    
    if phi_s <= 0:
        return []
    candidates = list(tm.keys())
    zero_score = [t for t in candidates if ts_needed.get(t, 0) == 0]
    freed = sum(instance.tool_sizes[t] for t in zero_score)
    if freed >= phi_s:
        chosen, acc = [], 0
        for t in zero_score:
            chosen.append(t)
            acc += instance.tool_sizes[t]
            if acc >= phi_s:
                break
        return chosen
    remove = list(zero_score)
    remaining = phi_s - freed
    valued = [t for t in candidates if ts_needed.get(t, 0) > 0]
    if HAS_PULP:
        prob = pulp.LpProblem("TRM", pulp.LpMinimize)
        lam = {t: pulp.LpVariable(f"l_{t}", cat="Binary") for t in valued}
        prob += pulp.lpSum(ts_needed.get(t,0)*lam[t] for t in valued)
        prob += pulp.lpSum(instance.tool_sizes[t]*lam[t] for t in valued) >= remaining
        pulp.PULP_CBC_CMD(msg=0).solve(prob)
        remove += [t for t in valued if pulp.value(lam[t]) > 0.5]
    else:
        acc = 0
        for t in sorted(valued, key=lambda t: ts_needed.get(t, 0)):
            remove.append(t)
            acc += instance.tool_sizes[t]
            if acc >= remaining:
                break
    return remove


def compute_trm(magazine, required_tool, succeeding_tools, instance):
    """Computes the TRM eviction set. Returns a list of tools to delete from the magazine to make room 
    for the required tool, based on the succeeding tools and the instance parameters.
    
    magazine: current magazine (tool -> size)
    required_tool: the tool we need to load
    succeeding_tools: list of tools needed in the future on this machine
    instance: the problem instance (for tool sizes and capacity)
    """
    
    C = instance.magazine_capacity
    phi_req = instance.tool_sizes[required_tool]
    phi_s = phi_req - (C - sum(magazine.values()))
    if phi_s <= 0:
        return []
    ts_needed_set = set(succeeding_tools) - {required_tool}
    score: Dict[int,int] = {}
    seen: set = set()
    ordered: List[int] = []
    for t in succeeding_tools:
        if t in ts_needed_set and t not in seen:
            ordered.append(t)
            seen.add(t)
    for u, t in enumerate(ordered):
        score[t] = len(ordered) - u
    return _trm_ilp(dict(magazine), score, phi_s, instance)


def _preinit_magazine(machine_tools, tool_sizes, C):
    """Pre-initializes the magazine with tools that are needed in the future, up to capacity C. 
    This is a heuristic to give the TRM a better starting point.
    
    machine_tools: list of tools needed in the future on this machine
    tool_sizes: dict of tool -> size
    C: magazine capacity
    """
    mag: Dict[int,int] = {}
    for t in machine_tools:
        if t not in mag:
            sz = tool_sizes.get(t, 0)
            if sum(mag.values()) + sz <= C:
                mag[t] = sz
    return mag


def evaluate(chrom: Chromosome, instance: Instance) -> float:
    """Evaluates the fitness of a chromosome by simulating the schedule and applying the TRM 
    for tool evictions. Returns the total weighted tardiness plus tool switch costs.
    
    chrom: the chromosome to evaluate
    instance: the problem instance
    """
    VI, VM = chrom
    seq = decode_sequence(VI, instance)
    τ, wd, ws, C = (instance.tool_setup_time, instance.wd,
                    instance.ws, instance.magazine_capacity)
    machine_seq: Dict[int, List[int]] = {m: [] for m in instance.machines}
    for g, op in enumerate(seq):
        machine_seq[VM[g]].append(op.tool_set)
    magazine = {m: _preinit_magazine(machine_seq[m], instance.tool_sizes, C)
                for m in instance.machines}
    avail: Dict[int,float] = {m: 0.0 for m in instance.machines}
    machine_pos: Dict[int,int] = {m: 0 for m in instance.machines}
    last_end: Dict[int,float] = {}
    fitness = 0.0
    for g, op in enumerate(seq):
        m, t_req = VM[g], op.tool_set
        pos = machine_pos[m]; machine_pos[m] += 1
        succeeding = machine_seq[m][pos+1:]
        mag = magazine[m]; switch = 0
        if t_req not in mag:
            switch = 1
            for t in compute_trm(mag, t_req, succeeding, instance):
                del mag[t]
            mag[t_req] = instance.tool_sizes[t_req]
        earliest = max(op.release_time, avail[m])
        if op.op_idx > 1:
            earliest = max(earliest, last_end.get(op.job_id, 0.0))
        end = earliest + op.proc_time + τ * switch
        avail[m] = end; last_end[op.job_id] = end
        fitness += wd * max(0.0, end - op.due_date) + ws * τ * switch
    return fitness


def evaluate_detailed(chrom: Chromosome, instance: Instance):
    """Evaluates the chromosome and also returns a breakdown of fitness contributions by machine. 

    Returns a tuple of (total fitness, dict of machine -> fitness contribution).

    chrom: the chromosome to evaluate
    instance: the problem instance
    """
    VI, VM = chrom
    seq = decode_sequence(VI, instance)
    τ, wd, ws, C = (instance.tool_setup_time, instance.wd,
                    instance.ws, instance.magazine_capacity)
    avail: Dict[int,float] = {m: 0.0 for m in instance.machines}
    machine_seq: Dict[int,List[int]] = {m: [] for m in instance.machines}
    for g, op in enumerate(seq):
        machine_seq[VM[g]].append(op.tool_set)
    magazine = {m: _preinit_magazine(machine_seq[m], instance.tool_sizes, C)
                for m in instance.machines}
    machine_pos: Dict[int,int] = {m: 0 for m in instance.machines}
    last_end: Dict[int,float] = {}
    fitness = 0.0
    machine_fitness: Dict[int,float] = {m: 0.0 for m in instance.machines}
    for g, op in enumerate(seq):
        m, t_req = VM[g], op.tool_set
        pos = machine_pos[m]; machine_pos[m] += 1
        succeeding = machine_seq[m][pos+1:]
        mag = magazine[m]; switch = 0
        if t_req not in mag:
            switch = 1
            for t in compute_trm(mag, t_req, succeeding, instance):
                del mag[t]
            mag[t_req] = instance.tool_sizes[t_req]
        earliest = max(op.release_time, avail[m])
        if op.op_idx > 1:
            earliest = max(earliest, last_end.get(op.job_id, 0.0))
        end = earliest + op.proc_time + τ * switch
        avail[m] = end; last_end[op.job_id] = end
        contrib = wd*max(0.0,end-op.due_date) + ws*τ*switch
        fitness += contrib; machine_fitness[m] += contrib
    return fitness, machine_fitness