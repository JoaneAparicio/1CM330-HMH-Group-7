"""
MEMETIC ALGORITHM - FINAL (paper-faithful MH + LS branch)
==========================================================

Architecture (per generation), exactly Dang et al. (2021) Algorithm 1
plus ONE added branch:

  Pk ── tournament ──→ CX/POX (phase logic) → mutation ──→ C'k   (CR+MUT offspring)
  Pk ── tournament ──→ LS-descent (until no improvement) → Lk    (LS offspring, NEW)
                evaluate → elitism selection + immigration (paper 5.5)

Paper components kept verbatim:
  - Tournament selection (gamma1 = 0.20, paper Sec. 7.2)
  - POX/CX phase switching: POX when best=true or q <= B (B=1), else CX
  - Mutation applied to ALL crossover offspring (Ps = Pu = 0.01)
  - Elitism (gamma2 = 0.10) + immigration (duplicates replaced by randoms)
  - Initial population: 1 PH chromosome + (Np-1) random

The ONLY addition: an LS branch that tournament-selects parents from Pk
(distinct within a generation) and descends each until no improving move
is found (true local search). Total offspring per generation = Np:
  (1 - LS_SHARE)*Np from CR->MUT  and  LS_SHARE*Np from LS.

NFC: every evaluate() call is counted (init, offspring, LS descents,
immigrants). Stop when NFC budget is reached.

Runs: base cases (Table 14) + Table 8 sub-instances of 6M140
(ops sorted by release time, first n taken).

Edit CONFIGURATION and press F5!
"""

import csv as _csv
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import random

try:
    from models import Instance, Chromosome, init_random_chromosome
    from evaluation import evaluate
    from ga_operators import (
        combined_crossover, problem_oriented_crossover,
        mutate, tournament_select, practitioner_heuristic
    )
    from local_search import local_search_vm
    from data_loader import load_instance_csv
    from export import save_memetic_excel
except ImportError as e:
    print(f"ERROR: Missing module - {e}")
    raise


# ============================================================================
# ✏️  CONFIGURATION - EDIT HERE
# ============================================================================

# ── What to run ─────────────────────────────────────────────────────────
#BASE_CASES_TO_RUN = ["2M38", "2M46", "6M140", "6M163"]   # [] to skip
BASE_CASES_TO_RUN = []
RUN_TABLE8 = True
TABLE8_CASE = "6M140"
TABLE8_N_VALUES = [15, 25, 30, 60, 90, 120, 140]

# ── Budget / runs ───────────────────────────────────────────────────────
NFC_BUDGET = 1650 # evaluation budget (match with friend's MH!)
RUNS = 10                  # runs per configuration
MAX_GENERATIONS = 1000
GC = 100                   # no-improvement generations to stop (match MH runner)

# ── Population & LS branch ──────────────────────────────────────────────
POPULATION_SIZE = 100      # Np (paper: 100)
LS_SHARE = 1/3             # fraction of offspring produced by the LS branch
LS_INTENSITY = 1           # ils_iter per LS round
LS_MAX_ROUNDS = 50         # safety cap on the descent loop

# ── Paper parameters (Dang et al. 2021, Section 7.2 tuning) ─────────────
B_PHASE = 1                # POX applied B times after a new best
GAMMA1 = 0.20              # tournament selection rate -> ST = gamma1 * Np
GAMMA2 = 0.10              # elitism selection rate    -> SE = gamma2 * Np
PS = 0.01                  # swap mutation probability (job vector)
PU = 0.01                  # uniform mutation probability (machine vector)

RANDOM_SEED = 42           # each run uses SEED + run_id
VERBOSE = False

# ── Instance metadata (for Excel export) ────────────────────────────────
INSTANCE_META = {
    "2M38":  {"n_machines": 2, "n_tools": 19, "capacity": 80},
    "2M46":  {"n_machines": 2, "n_tools": 23, "capacity": 80},
    "6M140": {"n_machines": 6, "n_tools": 70, "capacity": 80},
    "6M163": {"n_machines": 6, "n_tools": 82, "capacity": 80},
}

# ============================================================================
# DATA PATHS
# ============================================================================

HERE = Path.cwd()
BASE = HERE / "data"
TABLE8_DIR = HERE / "table8_instances"
OUTPUT_DIR = HERE / "output"

BASE_CASES = {
    "2M38":  str(BASE / "2M38" / "2M38.csv"),
    "2M46":  str(BASE / "2M46" / "2M46.csv"),
    "6M140": str(BASE / "6M140" / "6M140.csv"),
    "6M163": str(BASE / "6M163" / "6M163.csv"),
}


# ============================================================================
# TABLE 8 SUB-INSTANCE GENERATOR (lecturer's construction)
# ============================================================================

def make_table8_csv(base_csv: str, n: int, out_dir: Path) -> str:
    """Sort ops by release time (non-decreasing, stable), take first n."""
    with open(base_csv, newline="") as f:
        rows = list(_csv.reader(f))

    meta_rows, header_row, data_rows = [], None, []
    section = "meta"
    for row in rows:
        if not row or not row[0].strip():
            continue
        key = row[0].strip()
        if section == "meta":
            if key == "Job":
                header_row = row
                section = "data"
            else:
                meta_rows.append(row)
        else:
            try:
                float(row[2])
                data_rows.append(row)
            except (ValueError, IndexError):
                continue

    if n > len(data_rows):
        raise ValueError(f"n={n} > number of operations ({len(data_rows)})")

    data_rows.sort(key=lambda r: float(r[2]))
    sub_rows = data_rows[:n]
    new_meta = [["O", str(n)] if r[0].strip() == "O" else r for r in meta_rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(base_csv).stem}_n{n}.csv"
    with open(out_path, "w", newline="") as f:
        w = _csv.writer(f)
        for r in new_meta:
            w.writerow(r)
        if header_row:
            w.writerow(header_row)
        for r in sub_rows:
            w.writerow(r)
    return str(out_path)


# ============================================================================
# MEMETIC ALGORITHM (paper MH + LS branch)
# ============================================================================

def _key(chrom) -> Tuple:
    return (tuple(chrom[0]), tuple(chrom[1]))


class MemeticAlgorithm:

    def __init__(self, instance, pop_size=100, max_generations=1000,
                 nfc_budget=5000, gc=100, ls_share=1/3, ls_intensity=1,
                 ls_max_rounds=50, verbose=False):
        self.instance = instance
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.nfc_budget = nfc_budget
        self.gc = gc
        self.ls_count_target = max(1, int(round(ls_share * pop_size)))
        self.cr_count_target = pop_size - self.ls_count_target
        self.ls_intensity = ls_intensity
        self.ls_max_rounds = ls_max_rounds
        self.verbose = verbose

        self.ST = max(2, int(GAMMA1 * pop_size))   # tournament size (paper)
        self.SE = max(1, int(GAMMA2 * pop_size))   # number of elites (paper)

        self.start_time = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.nfc_count = 0

        self.stats = {'generation': 0, 'time': 0.0, 'nfc': 0,
                      'ls_descents': 0, 'ls_rounds': 0, 'pox_gens': 0}

    # ── basics ───────────────────────────────────────────────────────────
    def _eval(self, sol) -> float:
        """Budget-aware evaluation; every call counts toward NFC."""
        if self.nfc_count >= self.nfc_budget:
            return float('inf')
        self.nfc_count += 1
        return evaluate(sol, self.instance)

    def _tselect(self, population, fitnesses):
        """Tournament selection (paper). Robust to index- or chromosome-return."""
        sel = tournament_select(population, fitnesses, self.ST)
        if isinstance(sel, (int, np.integer)):
            return population[sel]
        return sel

    # ── CR -> MUT pipeline (paper lines 7-12) ───────────────────────────
    def _create_cr_mut_offspring(self, population, fitnesses, use_pox, count):
        offspring = []
        while len(offspring) < count:
            pa = self._tselect(population, fitnesses)
            pb = self._tselect(population, fitnesses)
            if use_pox:
                c1, c2 = problem_oriented_crossover(pa, pb, self.instance)
            else:
                c1, c2 = combined_crossover(pa, pb, self.instance)
            offspring.append(c1)
            if len(offspring) < count:
                offspring.append(c2)
        # Mutation on ALL crossover offspring (paper line 12)
        return [mutate(c, self.instance, PS, PU) for c in offspring]

    # ── LS branch (the added memetic component) ──────────────────────────
    def _ls_descend(self, solution, start_fitness):
        """True local search: accept ONLY improving moves, stop at local optimum."""
        current, current_fit = solution, start_fitness
        for _ in range(self.ls_max_rounds):
            if self.nfc_count >= self.nfc_budget:
                break
            candidate = local_search_vm(current, self.instance,
                                        ils_iter=self.ls_intensity)
            cand_fit = self._eval(candidate)
            self.stats['ls_rounds'] += 1
            if cand_fit < current_fit - 1e-6:
                current, current_fit = candidate, cand_fit
            else:
                break
        return current, current_fit

    def _create_ls_offspring(self, population, fitnesses):
        """Tournament-select distinct parents from Pk, descend each."""
        fit_map = {}
        for sol, fit in zip(population, fitnesses):
            fit_map.setdefault(_key(sol), fit)

        offspring, off_fits = [], []
        chosen = set()
        attempts = 0
        while len(offspring) < self.ls_count_target and attempts < 5 * self.ls_count_target:
            attempts += 1
            if self.nfc_count >= self.nfc_budget:
                break
            parent = self._tselect(population, fitnesses)
            k = _key(parent)
            if k in chosen and attempts < 3 * self.ls_count_target:
                continue   # prefer distinct starting points
            chosen.add(k)
            start_fit = fit_map.get(k)
            if start_fit is None:
                start_fit = self._eval(parent)
            improved, fit = self._ls_descend(parent, start_fit)
            offspring.append(improved)
            off_fits.append(fit)
            self.stats['ls_descents'] += 1
        return offspring, off_fits

    # ── Elitism selection + immigration (paper 5.5) ──────────────────────
    def _elitism_immigration(self, parents, parent_fits, offspring, off_fits):
        # SE best parents
        order = np.argsort(parent_fits)[:self.SE]
        best_parents = [parents[i] for i in order]
        best_fits = [parent_fits[i] for i in order]

        new_pop = list(offspring)
        new_fits = list(off_fits)

        # pad if LS branch was cut short by the NFC budget
        bi = 0
        sorted_idx = list(np.argsort(parent_fits))
        while len(new_pop) < self.pop_size and bi < len(sorted_idx):
            new_pop.append(parents[sorted_idx[bi]])
            new_fits.append(parent_fits[sorted_idx[bi]])
            bi += 1

        # replace SE randomly chosen offspring with the SE best parents
        repl = random.sample(range(len(new_pop)), min(self.SE, len(new_pop)))
        for slot, bp, bf in zip(repl, best_parents, best_fits):
            new_pop[slot] = bp
            new_fits[slot] = bf

        # immigration: replace duplicates with fresh random chromosomes
        seen = set()
        for i in range(len(new_pop)):
            k = _key(new_pop[i])
            if k in seen:
                newc = init_random_chromosome(self.instance)
                new_pop[i] = newc
                new_fits[i] = self._eval(newc)
            else:
                seen.add(k)

        return new_pop[:self.pop_size], new_fits[:self.pop_size]

    # ── main loop (paper Algorithm 1 + LS branch) ────────────────────────
    def run(self):
        self.start_time = time.time()

        # Initialization: 1 PH chromosome + (Np-1) random (paper 5.2)
        ph_chrom, _ = practitioner_heuristic(self.instance)
        population = [ph_chrom] + [init_random_chromosome(self.instance)
                                   for _ in range(self.pop_size - 1)]
        fitnesses = [self._eval(c) for c in population]

        self.best_fitness = min(fitnesses)
        self.best_solution = population[int(np.argmin(fitnesses))]

        best_flag = False
        q = B_PHASE + 1          # first generation uses CX (FIX-1)
        no_improve = 0

        while (self.nfc_count < self.nfc_budget
               and self.stats['generation'] < self.max_generations
               and no_improve < self.gc):

            self.stats['generation'] += 1
            use_pox = best_flag or (q <= B_PHASE)
            if use_pox:
                self.stats['pox_gens'] += 1

            # 1. CR -> MUT pipeline (paper)
            cr_offspring = self._create_cr_mut_offspring(
                population, fitnesses, use_pox, self.cr_count_target)
            cr_fits = [self._eval(c) for c in cr_offspring]

            # 2. LS branch from Pk (the added component)
            ls_offspring, ls_fits = self._create_ls_offspring(population, fitnesses)

            # 3. Elitism selection + immigration (paper)
            all_off = cr_offspring + ls_offspring
            all_off_fits = cr_fits + ls_fits
            population, fitnesses = self._elitism_immigration(
                population, fitnesses, all_off, all_off_fits)

            # 4. Phase / best bookkeeping (paper lines 16-24)
            fk = min(fitnesses)
            if fk < self.best_fitness - 1e-6:
                self.best_fitness = fk
                self.best_solution = population[int(np.argmin(fitnesses))]
                best_flag = True
                q = 1
                no_improve = 0
            else:
                best_flag = False
                q += 1
                no_improve += 1

            if self.verbose and (self.stats['generation'] % 5 == 0):
                print(f"Gen {self.stats['generation']:3d} "
                      f"[{'POX' if use_pox else 'CX '}]: "
                      f"Fit={self.best_fitness:8.2f}  NFC={self.nfc_count:6d}  "
                      f"Time={time.time()-self.start_time:7.1f}s")

        self.stats['time'] = time.time() - self.start_time
        self.stats['nfc'] = self.nfc_count
        return self.best_solution, self.best_fitness, self.stats


# ============================================================================
# RUN ONE CONFIGURATION
# ============================================================================

def run_configuration(label: str, instance, case_key: str = "") -> Dict[str, Any]:
    print(f"{'='*80}")
    print(f"CONFIG: {label}")
    print(f"{'='*80}")
    print(f"✓ {instance.n_ops} ops, {instance.n_machines} machines\n")

    print(f"Computing PH baseline ({RUNS} runs)...")
    ph_results = []
    for run_id in range(RUNS):
        random.seed(run_id)
        _, ph_fitness = practitioner_heuristic(instance)
        ph_results.append(ph_fitness)
    ph_fit = float(np.mean(ph_results))
    ph_std = float(np.std(ph_results))
    print(f"PH: {ph_fit:.4f} ± {ph_std:.4f}\n")

    print(f"Running Memetic ({RUNS} runs)...")
    run_records = []
    for run_id in range(1, RUNS + 1):
        np.random.seed(RANDOM_SEED + run_id)
        random.seed(RANDOM_SEED + run_id)

        ma = MemeticAlgorithm(
            instance,
            pop_size=POPULATION_SIZE,
            max_generations=MAX_GENERATIONS,
            nfc_budget=NFC_BUDGET,
            gc=GC,
            ls_share=LS_SHARE,
            ls_intensity=LS_INTENSITY,
            ls_max_rounds=LS_MAX_ROUNDS,
            verbose=VERBOSE
        )
        solution, fitness, stats = ma.run()
        run_records.append({'fitness': fitness, 'nfc': stats['nfc'],
                            'time': stats['time'], 'gens': stats['generation']})

        arpd = 100 * (ph_fit - fitness) / ph_fit if ph_fit > 0 else 0.0
        print(f"  Run {run_id}: {fitness:.4f}  NFC: {stats['nfc']:6d}  "
              f"Gens: {stats['generation']:4d}  Time: {stats['time']:6.1f}s  "
              f"ARPD: {arpd:+.2f}%")

    fits = [r['fitness'] for r in run_records]
    mean_fit, std_fit = float(np.mean(fits)), float(np.std(fits))
    best_fit, worst_fit = float(np.min(fits)), float(np.max(fits))
    arpd_mean = 100 * (ph_fit - mean_fit) / ph_fit if ph_fit > 0 else 0.0
    total_time = sum(r['time'] for r in run_records)
    avg_time   = total_time / RUNS
    avg_gens   = float(np.mean([r['gens'] for r in run_records]))

    print(f"  Mean: {mean_fit:.4f} ± {std_fit:.4f}  Best: {best_fit:.4f}  "
          f"Worst: {worst_fit:.4f}  ARPD: {arpd_mean:+.2f}%")
    print(f"  Config time: {total_time/60:.1f} min\n")

    # Metadata for the Excel detail sheet
    meta = INSTANCE_META.get(case_key, {})
    run_details = [(r['fitness'], r['time'], r['gens']) for r in run_records]

    return {
        'label': label, 'n_ops': instance.n_ops,
        'n_machines': meta.get("n_machines", instance.n_machines),
        'n_tools':    meta.get("n_tools",    getattr(instance, "n_tools", "—")),
        'capacity':   meta.get("capacity",   getattr(instance, "capacity", "—")),
        'ph_mean': ph_fit, 'ph_std': ph_std,
        'mean': mean_fit, 'std': std_fit,
        'best': best_fit, 'worst': worst_fit,
        'arpd_mean': arpd_mean,
        'total_time': total_time, 'avg_time': avg_time, 'avg_gens': avg_gens,
        'runs': fits,
        'run_details': run_details,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print(f"MEMETIC FINAL: paper MH (tournament + POX/CX + mutation + elitism)")
    print(f"               + LS branch from Pk (tournament-selected, descent)")
    print(f"{'='*80}")
    print(f"Base cases: {BASE_CASES_TO_RUN if BASE_CASES_TO_RUN else 'none'}  |  "
          f"Table 8: {'ON' if RUN_TABLE8 else 'OFF'}")
    print(f"NFC: {NFC_BUDGET:,}  |  Runs: {RUNS}  |  Np: {POPULATION_SIZE}  |  "
          f"LS share: {LS_SHARE:.2f}  |  Gc: {GC}")
    print(f"Paper params: B={B_PHASE}, gamma1={GAMMA1}, gamma2={GAMMA2}, "
          f"Ps={PS}, Pu={PU}")
    print(f"{'='*80}\n")

    all_results = []

    for case in BASE_CASES_TO_RUN:
        csv_path = BASE_CASES.get(case)
        if csv_path is None or not Path(csv_path).exists():
            print(f"⚠️  {case}: file not found, skipping\n")
            continue
        instance = load_instance_csv(csv_path)
        result = run_configuration(f"{case}  ({case[:2]}-{case[2:]}ops)", instance, case_key=case)
        result['table'] = 14
        all_results.append(result)

    if RUN_TABLE8:
        base_csv = BASE_CASES.get(TABLE8_CASE)
        if base_csv is None or not Path(base_csv).exists():
            print(f"⚠️  Table 8 base case {TABLE8_CASE} not found, skipping\n")
        else:
            print(f"{'='*80}")
            print(f"TABLE 8: sub-instances of {TABLE8_CASE} "
                  f"(sorted by release time, first n ops)")
            print(f"{'='*80}\n")
            for n in TABLE8_N_VALUES:
                try:
                    sub_csv = make_table8_csv(base_csv, n, TABLE8_DIR)
                except ValueError as e:
                    print(f"⚠️  n={n}: {e}, skipping")
                    continue
                instance = load_instance_csv(sub_csv)
                result = run_configuration(f"{TABLE8_CASE} n={n}", instance, case_key=TABLE8_CASE)
                result['table'] = 8
                all_results.append(result)

    # ── Summary console print ────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"SUMMARY TABLE")
    print(f"{'='*100}\n")
    print(f"{'Config':<18} {'Table':<6} {'PH Mean':<10} {'Mean':<10} {'Std':<9} "
          f"{'Best':<9} {'Worst':<9} {'ARPD%':<8}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['label']:<18} {r['table']:<6} {r['ph_mean']:<10.4f} "
              f"{r['mean']:<10.4f} {r['std']:<9.4f} {r['best']:<9.4f} "
              f"{r['worst']:<9.4f} {r['arpd_mean']:<+8.2f}")

    print(f"\n{'='*100}")
    print(f"PER-RUN VALUES (for box plots / compare_stats.py):")
    print(f"{'='*100}\n")
    for r in all_results:
        print(f"{r['label']}: {[round(v, 4) for v in r['runs']]}")

    grand_total = sum(r['total_time'] for r in all_results)
    print(f"\n{'='*100}")
    print(f"TIMING SUMMARY:")
    print(f"  Configurations: {len(all_results)}")
    print(f"  Runs per config: {RUNS}")
    print(f"  Total runs: {len(all_results) * RUNS}")
    print(f"  Total elapsed time: {grand_total/60:.1f} minutes "
          f"({grand_total/3600:.2f} hours)")
    print(f"{'='*100}\n")

    # ── Excel export ─────────────────────────────────────────────────────
    if all_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stop_str = f"evals={NFC_BUDGET}"

        def _make_meta(case_label):
            path = str(OUTPUT_DIR / f"results_memetic_{case_label}.xlsx")
            return {
                "run_name":  path,
                "timestamp": datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                "stop_str":  stop_str,
                "params": {
                    "stop":     "evals",
                    "evals":    NFC_BUDGET,
                    "pop":      POPULATION_SIZE,
                    "gc":       GC,
                    "runs":     RUNS,
                    "seed":     RANDOM_SEED,
                    "ls_share": f"{LS_SHARE:.2f}",
                    "case":     case_label,
                },
            }

        base_results   = [r for r in all_results if r.get("table") == 14]
        table8_results = [r for r in all_results if r.get("table") == 8]

        if base_results:
            out_base = str(OUTPUT_DIR / f"results_memetic_base.xlsx")
            save_memetic_excel(base_results, output_path=out_base,
                               meta=_make_meta("base"))

        if table8_results:
            out_t8 = str(OUTPUT_DIR / f"results_memetic_table8.xlsx")
            save_memetic_excel(table8_results, output_path=out_t8,
                               meta=_make_meta("table8"))


if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
