"""
ga_operators.py – GA operators: crossover, mutation, selection, PH.

Implements every genetic algorithm operator plus the main GA loop. 
The main loop is implemented in the matheuristic() function, which takes an instance and GA parameters 
as input and returns a RunResult object containing the best fitness, best chromosome, fitness history, 
computation time, and number of generations.
"""
from __future__ import annotations
import random
import copy
import time
from dataclasses import dataclass, field as _field
from typing import List, Dict, Optional
from dataclasses import dataclass as _dc

from models import Instance, Chromosome, init_random_chromosome
from evaluation import evaluate


# ── Crossover ─────────────────────────────────────────────────────────────

def two_point_crossover_VM(vm1, vm2):
    """Simple two-point crossover for the VM part of the chromosome. Returns two offspring VM lists.
    
    vm1, vm2: parent VM lists
    """
    n = len(vm1)
    p1, p2 = sorted(random.sample(range(n), 2))
    return vm1[:p1]+vm2[p1:p2]+vm1[p2:], vm2[:p1]+vm1[p1:p2]+vm2[p2:]


def apmx(vi1, vi2):
    """The Adapted Partial Mapped Crossover (APMX) for the job-order vector VI. Returns two offspring VI lists.

    vi1, vi2: parent VI lists
    """
    n = len(vi1)
    tp1 = list(range(1, n+1))
    pos_map1: Dict[int, List[int]] = {}
    for g, job in enumerate(vi1):
        pos_map1.setdefault(job, []).append(g+1)
    occ2: Dict[int,int] = {}
    tp2: List[int] = []
    for job in vi2:
        k = occ2.get(job, 0)
        tp2.append(pos_map1[job][k])
        occ2[job] = k+1
    p1, p2 = sorted(random.sample(range(n), 2))
    po1 = tp1[:p1]+tp2[p1:p2]+tp1[p2:]
    po2 = tp2[:p1]+tp1[p1:p2]+tp2[p2:]
    map_21 = {}; map_12 = {}
    for a, b in zip(tp1[p1:p2], tp2[p1:p2]):
        map_12[a] = b; map_21[b] = a

    def legalize(po, forbidden, mapping):
        """Legalize the offspring by replacing forbidden values using the mapping until a legal value is found.
        
        po: the offspring list to legalize
        forbidden: the set of values that are not allowed in the current positions (the crossover window
        mapping: the mapping to use for replacement (map_12 or map_21 depending on the offspring)
        """
        result = list(po)
        for i in list(range(p1))+list(range(p2, n)):
            val = result[i]; seen: set = set()
            while val in forbidden:
                if val in seen: break
                seen.add(val); val = mapping.get(val, val)
            result[i] = val
        return result

    po1 = legalize(po1, set(tp2[p1:p2]), map_21)
    po2 = legalize(po2, set(tp1[p1:p2]), map_12)
    pos_to_job1 = {g+1: job for g, job in enumerate(vi1)}
    return [pos_to_job1[s] for s in po1], [pos_to_job1[s] for s in po2]


def combined_crossover(p1, p2, instance):
    """Combined crossover: APMX for VI + two-point crossover for VM. Returns two offspring chromosomes
    
    p1, p2: parent 1 and 2 chromosome (VI, VM)
    """
    vi1,vm1 = p1; vi2,vm2 = p2
    nvi1,nvi2 = apmx(vi1,vi2)
    nvm1,nvm2 = two_point_crossover_VM(vm1,vm2)
    return (nvi1,nvm1),(nvi2,nvm2)


def constructive_heuristic(vi, instance):
    """Constructive heuristic to build a VM list from a given VI list, following the logic of the practitioner heuristic 
    but without the initial tool allocation phase. Returns a VM list.
    
    vi: the job-order vector for which to construct the VM list
    instance: the problem instance, used to access job and operation data for machine assignment
    """
    C = instance.magazine_capacity
    TM: Dict[int,Dict[int,int]] = {m:{} for m in instance.machines}
    pM: Dict[int,float] = {m:0.0 for m in instance.machines}
    counters: Dict[int,int] = {jid:0 for jid in instance.jobs}
    VM: List[int] = []
    for job_id in vi:
        k = counters[job_id]; counters[job_id] += 1
        op = instance.jobs[job_id][k]
        t_req = op.tool_set; phi_req = instance.tool_sizes[t_req]
        mwt = [m for m,mag in TM.items() if t_req in mag]
        if mwt:
            # tie-break: lowest index among machines with equal load
            m_star = min(mwt, key=lambda m: (pM[m], m))
        else:
            MC = [m for m,mag in TM.items() if C-sum(mag.values())>=phi_req]
            if MC:
                m_star = min(MC, key=lambda m: (pM[m], m))
                TM[m_star][t_req] = phi_req
            else:
                m_star = min(instance.machines, key=lambda m: (pM[m], m))
        pM[m_star] += op.proc_time; VM.append(m_star)
    return VM


def edd_rearrange(vi, p1_cut, p2_cut, instance):
    """Rearrange the jobs in the positions outside the [p1_cut, p2_cut) window according to EDD order, 
    while keeping the jobs inside the window fixed. Returns a new VI list.
    Used in the problem-oriented crossover to reorder the offspring VI according to EDD before applying the 
    constructive heuristic.
    
    vi: the offspring VI list to be rearranged
    p1_cut, p2_cut: the crossover cut points defining the window of fixed positions
    instance: the problem instance, used to access job and operation data for EDD ordering
    """

    positions_outside = list(range(p1_cut))+list(range(p2_cut, len(vi)))
    occ: Dict[int,int] = {}
    due_per_position = []
    for pos in positions_outside:
        job_id = vi[pos]; occ[job_id] = occ.get(job_id,0)+1
        op = instance.jobs[job_id][occ[job_id]-1]
        due_per_position.append((pos, op.due_date, job_id))
    due_per_position.sort(key=lambda x: x[1])
    new_vi = list(vi)
    for (orig_pos,_,_),(_, _,job_id) in zip(
            sorted([(p,d,j) for p,d,j in due_per_position], key=lambda x: x[0]),
            due_per_position):
        new_vi[orig_pos] = job_id
    return new_vi


def problem_oriented_crossover(p1, p2, instance):
    """Problem-oriented crossover: APMX for VI + EDD rearrangement + constructive heuristic. 
    Returns two offspring chromosomes.
    
    p1, p2: parent 1 and 2 chromosome (VI, VM)
    instance: the problem instance, used for EDD rearrangement and constructive heuristic
    """
    vi1,vm1 = p1; vi2,vm2 = p2
    n = len(vi1)
    cut1,cut2 = sorted(random.sample(range(n),2))
    o1_vi,o2_vi = apmx(vi1,vi2)
    ro1_vi = edd_rearrange(o1_vi,cut1,cut2,instance)
    ro2_vi = edd_rearrange(o2_vi,cut1,cut2,instance)
    return (ro1_vi,constructive_heuristic(ro1_vi,instance)), (ro2_vi,constructive_heuristic(ro2_vi,instance))


# ── Mutation ──────────────────────────────────────────────────────────────

def swap_mutation(vi, ps):
    """Swap mutation for the VI part of the chromosome. Each gene is swapped with another random gene 
    with probability ps. Returns a new VI list.
    
    vi: the parent VI list to mutate
    ps: the mutation probability for each gene
    """
    new_vi = list(vi)
    for g in range(len(new_vi)):
        if random.random() < ps:
            g2 = random.randrange(len(new_vi))
            new_vi[g],new_vi[g2] = new_vi[g2],new_vi[g]
    return new_vi


def uniform_mutation(vm, pu, machines):
    """Uniform mutation for the VM part of the chromosome. Each gene is replaced with a random machine 
    with probability pu. Returns a new VM list.
    
    vm: the parent VM list to mutate
    pu: the mutation probability for each gene
    machines: the list of available machines
    """
    new_vm = list(vm)
    for g in range(len(new_vm)):
        if random.random() < pu:
            new_vm[g] = random.choice(machines)
    return new_vm


def mutate(chrom, instance, ps, pu):
    """Mutate a chromosome by applying swap mutation to the VI and uniform mutation to the VM. 
    Returns a new chromosome.
    
    chrom: the parent chromosome (VI, VM) to mutate
    instance: the problem instance, used to access machine data for uniform mutation
    ps: the mutation probability for the VI genes
    pu: the mutation probability for the VM genes
    """
    vi,vm = chrom
    return (swap_mutation(vi,ps), uniform_mutation(vm,pu,instance.machines))


# ── Selection ─────────────────────────────────────────────────────────────

def tournament_select(population, fitness_vals, tournament_size):
    """Tournament selection: randomly sample 'tournament_size' individuals and return the one 
    with the best fitness value among them.
    
    population: the list of chromosomes in the current population
    fitness_vals: the list of fitness values corresponding to the population
    tournament_size: the number of individuals to sample for the tournament
    """
    contestants = random.sample(range(len(population)), tournament_size)
    return population[min(contestants, key=lambda i: fitness_vals[i])]


def elitism_selection(parents, parent_fits, offspring, offspring_fits,
                      np_size, elitism_rate, instance):
    """ Elitism selection: combine parents and offspring, keep the best 'SE' parents, and fill 
    the rest of the new population with offspring. Then perform deduplication.
    
    parents: the list of parent chromosomes
    parent_fits: the list of fitness values corresponding to the parents
    offspring: the list of offspring chromosomes
    offspring_fits: the list of fitness values corresponding to the offspring
    np_size: the population size (number of individuals to keep in the new population)
    elitism_rate: the proportion of the population to keep from the parents (e.g. 0.10 to keep the best 10%)
    instance: the problem instance, used for generating random chromosomes during deduplication
    """
    SE = max(1, int(elitism_rate*np_size))
    sorted_parents = sorted(zip(parent_fits,parents), key=lambda x: x[0])
    best_parents = [p for _,p in sorted_parents[:SE]]
    best_fits    = [f for f,_ in sorted_parents[:SE]]

    new_pop = list(offspring); new_fits = list(offspring_fits)

    # Replace SE randomly chosen offspring slots with the SE best parents
    replace_idx = random.sample(range(len(new_pop)), min(SE,len(new_pop)))
    elitism_positions = set()
    for i, idx in enumerate(replace_idx):
        if i < len(best_parents):
            new_pop[idx] = best_parents[i]
            new_fits[idx] = best_fits[i]
            elitism_positions.add(idx)   # [FIX-4] track positions holding elites

    # Immigration: replace duplicates with random chromosomes.
    seen = set()
    for i in range(len(new_pop)):
        key = (tuple(new_pop[i][0]), tuple(new_pop[i][1]))
        if key in seen and i not in elitism_positions:
            new_pop[i] = init_random_chromosome(instance)
            new_fits[i] = evaluate(new_pop[i], instance)
        else:
            seen.add(key)
    return new_pop, new_fits


# ── Practitioner Heuristic ────────────────────────────────────────────────

def practitioner_heuristic(instance, threshold=72.0):
    """
    Two-phase heuristic (Algorithm 4 of Dang et al. 2021).

    Phase 1 – initial tool allocation to machines.
    Phase 2 – EDD-ordered operation assignment.

    The random element on line 20 of the paper (rand tool removal) is kept
    intentionally; callers that need a deterministic result should fix
    random.seed() before calling this function.
    """
    C, τ, wd, ws = (instance.magazine_capacity, instance.tool_setup_time,
                    instance.wd, instance.ws)
    ops_sorted = sorted(instance.operations, key=lambda o: o.due_date)
    TM: Dict[int,Dict[int,int]] = {m:{} for m in instance.machines}
    avail: Dict[int,float] = {m:0.0 for m in instance.machines}
    last_end: Dict[int,float] = {}

    # ── Phase 1: place tool sets into machines ──
    for op in ops_sorted:
        t_req = op.tool_set; phi_req = instance.tool_sizes[t_req]
        mwt = [m for m in instance.machines if t_req in TM[m]]
        if not mwt:
            MC = [m for m in instance.machines if C - sum(TM[m].values()) >= phi_req]
            if MC:
                # [FIX-2] tie-break: fewest tools first, then lowest machine index
                m_star = min(MC, key=lambda m: (len(TM[m]), m))
                TM[m_star][t_req] = phi_req

    # ── Phase 2: assign operations ──
    assignment = []; obj = 0.0
    for op in ops_sorted:
        t_req = op.tool_set; phi_req = instance.tool_sizes[t_req]

        # mP: earliest available machine (tie-break: lowest index)
        mP = min(instance.machines, key=lambda m: (avail[m], m))

        # mwt: all machines currently holding the required tool
        mwt = [m for m in instance.machines if t_req in TM[m]]

        # if multiple machines hold the tool, pick the one
        # with the earliest start time for this operation 
        def es(m):
            e = max(op.release_time, avail[m])
            if op.op_idx > 1:
                e = max(e, last_end.get(op.job_id, 0.0))
            return e

        if mwt:
            mT = min(mwt, key=lambda m: (es(m), m))
        else:
            mT = None

        # Trade-off: use mP (tool switch) vs mT (no switch)
        if mT is not None and mT != mP:
            if es(mT) - es(mP) >= threshold:
                m_star, switch = mP, 1   # mP is much earlier → pay switch cost
            else:
                m_star, switch = mT, 0   # mT is not much later → avoid switch
        elif mT is not None:
            m_star, switch = mT, 0       # mP == mT → no switch needed
        else:
            m_star, switch = mP, 1       # no machine has the tool → must switch

        # Perform tool switch: randomly remove tools until enough space freed
        if switch:
            phi_s = phi_req - (C - sum(TM[m_star].values()))
            if phi_s > 0:
                cands = list(TM[m_star].keys())
                random.shuffle(cands)          # paper explicitly uses rand()
                freed = 0
                for t in cands:
                    if freed >= phi_s:
                        break
                    del TM[m_star][t]
                    freed += instance.tool_sizes[t]
            TM[m_star][t_req] = phi_req

        end = es(m_star) + op.proc_time + τ * switch
        avail[m_star] = end
        last_end[op.job_id] = end
        obj += wd * max(0.0, end - op.due_date) + ws * τ * switch
        assignment.append((op, m_star))

    VI = [op.job_id for op, _ in assignment]
    VM = [m for _, m in assignment]
    return (VI, VM), obj


# ── GA parameters & main matheuristic loop ────────────────────────────────

@dataclass
class GAParams:
    """Parameters for the genetic algorithm matheuristic."""
    np_size:   int   = 100
    ps:        float = 0.01
    pu:        float = 0.01
    gamma1:    float = 0.20
    gamma2:    float = 0.10
    B:         int   = 1
    max_time:  float = 3600.0
    Gc:        int   = 20
    seed:      int   = 42
    max_evals: int   = 10_000   # budget for matheuristic_parallel_vm_ls


@_dc
class RunResult:
    """Result of a matheuristic run."""
    best_fitness:     float
    best_chromosome:  object
    history:          List[float]
    computation_time: float
    n_generations:    int


def matheuristic(instance, params):
    """Genetic algorithm matheuristic.

    instance: the problem instance to solve
    params: the GA parameters (see GAParams dataclass)
    Returns a RunResult object containing the best fitness, best chromosome, fitness history, 
    computation time, and number of generations.
    """
    random.seed(params.seed)
    start = time.time()
    Np  = params.np_size
    ST  = max(2, int(params.gamma1*Np))

    ph_chrom, _ = practitioner_heuristic(instance)
    population  = [ph_chrom] + [init_random_chromosome(instance) for _ in range(Np-1)]
    fitness_vals = [evaluate(c, instance) for c in population]
    fbest     = min(fitness_vals)
    best_idx  = fitness_vals.index(fbest)
    best_chrom = copy.deepcopy(population[best_idx])
    history = [fbest]; best = False; q = params.B + 1; no_improve = 0; gen = 0

    while True:
        if time.time() - start >= params.max_time: break
        if no_improve >= params.Gc: break
        gen += 1; offspring = []

        while len(offspring) < Np:
            pa = tournament_select(population, fitness_vals, ST)
            pb = tournament_select(population, fitness_vals, ST)
            # Paper Algorithm 1 lines 7-10:
            # if best = true OR q ≤ B  →  POX
            # else                     →  CX
            if best or q <= params.B:
                c1, c2 = problem_oriented_crossover(pa, pb, instance)
            else:
                c1, c2 = combined_crossover(pa, pb, instance)
            offspring.append(c1)
            if len(offspring) < Np:
                offspring.append(c2)

        offspring    = [mutate(c, instance, params.ps, params.pu) for c in offspring]
        off_fits     = [evaluate(c, instance) for c in offspring]
        population, fitness_vals = elitism_selection(
            population, fitness_vals, offspring, off_fits, Np, params.gamma2, instance)
        fk = min(fitness_vals)

        # Paper Algorithm 1 lines 17-24
        if fk < fbest:
            fbest     = fk
            best_idx  = fitness_vals.index(fk)
            best_chrom = copy.deepcopy(population[best_idx])
            best      = True    # line 19
            q         = 1       # line 20
            no_improve = 0
        else:
            best      = False   # line 22
            q        += 1       # line 23
            no_improve += 1

        history.append(fbest)

    return RunResult(fbest, best_chrom, history, time.time()-start, gen)


def matheuristic_ls(instance, params):
    """GA matheuristic followed by a Local Search post-processing step (MH+LS).

    Runs the standard matheuristic and then applies local_search() on the best
    chromosome found, using the remaining time budget if any is left, or a short
    fixed budget otherwise.  The LS uses ils_iter adapted to instance size
    (same logic as ql_matheuristic).

    instance: the problem instance to solve
    params: the GA parameters (see GAParams dataclass)
    Returns a RunResult with the (possibly improved) best fitness and updated time.
    """
    # Lazy import to avoid circular dependency (local_search imports evaluate/models)
    from local_search import local_search

    def _ils_iter_for(n_ops: int) -> int:
        if n_ops < 60:    return 3
        elif n_ops < 100: return 1
        else:             return 0

    import time as _time
    start_ls = _time.time()

    # Run the base MH
    result = matheuristic(instance, params)

    # Apply LS post-processing on the best chromosome found
    ils_iter = _ils_iter_for(instance.n_ops)
    ls_chrom = local_search(result.best_chromosome, instance, ils_iter=ils_iter)
    ls_fit   = evaluate(ls_chrom, instance)

    total_time = _time.time() - start_ls

    # Return an improved RunResult if LS found a better solution
    if ls_fit < result.best_fitness:
        new_history = result.history + [ls_fit]
        return RunResult(ls_fit, ls_chrom, new_history, total_time, result.n_generations)
    else:
        return RunResult(result.best_fitness, result.best_chromosome,
                         result.history, total_time, result.n_generations)


# ── Parallel MH+LS (VM-only) ──────────────────────────────────────────────

def matheuristic_parallel_vm_ls(instance, params):
    """MH+PLS faithful to the scheme: for each individual generates CR, MU and LS
    offspring in parallel and performs 4-way selection (parent, CR, MU, LS).

    The LS reuses the parent's already-known fitness (no re-evaluation),
    performs a single VM-relocate pass (first improvement) and exits.
    Cost per individual: same as CR + MU combined.
    """
    import time as _time, copy as _copy

    def _vm_relocate_pass(chrom, known_fit, instance):
        """One VM-relocate pass reusing the already-known fitness.
        Does not re-evaluate at the start — uses known_fit directly.
        Exits as soon as the first improving move is found (first improvement).
        Returns (new_chrom, new_fit) or (chrom, known_fit) if no improvement found.
        """
        VI, VM = list(chrom[0]), list(chrom[1])
        current_fit = known_fit
        improved = False
        for g in range(len(VI)):
            orig_m = VM[g]
            for m in instance.machines:
                if m == orig_m:
                    continue
                VM[g] = m
                new_fit = evaluate((VI, VM), instance)
                if new_fit < current_fit:
                    current_fit = new_fit
                    orig_m = m
                    improved = True
                    break        # first improvement: move to next individual
                VM[g] = orig_m
            if improved:
                break            # also exit the gene loop
        return (VI, VM), current_fit

    random.seed(params.seed)
    start      = _time.time()
    Np         = params.np_size
    ST         = max(2, int(params.gamma1 * Np))
    max_evals  = getattr(params, 'max_evals', 10_000)
    eval_count = 0
    best_flag  = False
    q          = params.B + 1

    # ── Initialize population ────────────────────────────────────────────
    ph_chrom, _ = practitioner_heuristic(instance)
    population   = [ph_chrom] + [init_random_chromosome(instance)
                                  for _ in range(Np - 1)]
    fitness_vals = []
    for c in population:
        fitness_vals.append(evaluate(c, instance))
        eval_count += 1

    fbest      = min(fitness_vals)
    best_idx   = fitness_vals.index(fbest)
    best_chrom = _copy.deepcopy(population[best_idx])
    history    = [fbest]
    no_improve = 0
    gen        = 0

    # ── Main loop ────────────────────────────────────────────────────────
    while True:
        if eval_count >= max_evals:                 break
        if _time.time() - start >= params.max_time: break
        if no_improve >= params.Gc:                 break

        gen      += 1
        offspring  = []
        off_fits   = []

        for idx in range(Np):
            if eval_count >= max_evals:
                break

            parent     = population[idx]
            parent_fit = fitness_vals[idx]

            # 1. CR child – crossover with partner tournament-selected from the population
            partner = tournament_select(population, fitness_vals, ST)
            if best_flag or q <= params.B:
                cx_child, _ = problem_oriented_crossover(parent, partner, instance)
            else:
                cx_child, _ = combined_crossover(parent, partner, instance)
            cx_fit    = evaluate(cx_child, instance)
            eval_count += 1

            # 2. MU child – mutation of the parent
            mut_child = mutate(parent, instance, params.ps, params.pu)
            mut_fit   = evaluate(mut_child, instance)
            eval_count += 1

            # 3. LS child – VM-relocate pass starting from the parent
            ls_child, ls_fit = _vm_relocate_pass(parent, parent_fit, instance)
            # Only counts 1 eval if there was an improvement (already evaluated inside the pass)
            # if ls_fit < parent_fit, 1 eval was used inside the pass; if ls_fit == parent_fit, no eval was used
            if ls_fit < parent_fit:
                eval_count += 1

            # 4-way selection: better than {parent, CR, MU, LS}
            candidates = [
                (parent_fit, parent),
                (cx_fit,     cx_child),
                (mut_fit,    mut_child),
                (ls_fit,     ls_child),
            ]
            best_fit_cand, best_cand = min(candidates, key=lambda x: x[0])
            offspring.append(best_cand)
            off_fits.append(best_fit_cand)

        if not offspring:
            break

        population, fitness_vals = elitism_selection(
            population, fitness_vals,
            offspring, off_fits,
            Np, params.gamma2, instance)

        fk = min(fitness_vals)
        if fk < fbest:
            fbest      = fk
            best_idx   = fitness_vals.index(fk)
            best_chrom = _copy.deepcopy(population[best_idx])
            no_improve = 0
            best_flag  = True
            q          = 1
        else:
            no_improve += 1
            best_flag  = False
            q         += 1

        history.append(fbest)

    return RunResult(fbest, best_chrom, history, _time.time() - start, gen)