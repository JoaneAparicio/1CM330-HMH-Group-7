"""
run_mh.py – Runs the base Matheuristic on one or all instances.

Budget controlled by time (--stop time) or function evaluations (--stop evals).

Use:
    python run_mh.py                             # Table 8 mode (default)
    python run_mh.py --runs 10 --stop time --time 3600   # exact paper replication
    python run_mh.py --runs 3  --stop time --time 120    # quick test
    python run_mh.py --case 2M38                 # only one base case
"""
# ── Imports ──────────────────────────────────────────────────────────────
from __future__ import annotations

from ga_operators import GAParams, practitioner_heuristic, matheuristic
from data_loader  import load_instance_csv
from export       import save_mh_only_excel
from models       import Instance
from pathlib      import Path
from datetime     import datetime
import argparse, os, sys, random


# ── Paths ─────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
BASE = HERE / "data"
OUT  = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ── Normal base cases ─────────────────────────────────────────────────────
BASE_CASES = {
    "2M38":  (f"{BASE}/2M38/2M38.csv",   "2M38  (2M-38ops)"),
    "2M46":  (f"{BASE}/2M46/2M46.csv",   "2M46  (2M-46ops)"),
    "6M140": (f"{BASE}/6M140/6M140.csv", "6M140 (6M-140ops)"),
    "6M163": (f"{BASE}/6M163/6M163.csv", "6M163 (6M-163ops)"),
}

# ── Table 8 sub-instances (Dang et al. 2021, §8.1) ────────────────────────
# All ops in 6M140 sorted by release_time; keep first n.  Maps n → label.
TABLE_8 = {
    15:  "6M140-n15  (6M-15ops)",
    25:  "6M140-n25  (6M-25ops)",
    30:  "6M140-n30  (6M-30ops)",
    60:  "6M140-n60  (6M-60ops)",
    90:  "6M140-n90  (6M-90ops)",
    120: "6M140-n120 (6M-120ops)",
    140: "6M140-n140 (6M-140ops)",
}


def _make_subinstance(full: Instance, n: int) -> Instance:
    """First *n* ops of 6M140 sorted by release_time."""
    sorted_ops = sorted(full.operations, key=lambda o: o.release_time)
    sub_ops    = sorted_ops[:n]
    used_tools = {op.tool_set for op in sub_ops}
    sub_tools  = {t: s for t, s in full.tool_sizes.items() if t in used_tools}
    return Instance(
        operations=sub_ops,
        machines=full.machines,
        tool_sizes=sub_tools,
        magazine_capacity=full.magazine_capacity,
        tool_setup_time=full.tool_setup_time,
    )


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MH only runner")
    p.add_argument("--runs",   type=int,   default=10,
                   help="Runs per instance (paper uses 10)")
    p.add_argument("--pop",    type=int,   default=100,  help="Population size")
    p.add_argument("--gc",     type=int,   default=100,  help="No-improve generations to stop")
    p.add_argument("--seed",   type=int,   default=42,   help="Base random seed")
    p.add_argument("--case",   type=str,   default="table8",
                   help="table8 (default) | 2M38 | 2M46 | 6M140 | 6M163 | all")
    p.add_argument("--output", type=str,   default=None, help="Output Excel filename")
    p.add_argument("--name",   type=str,   default="",   help="Experiment name for Excel metadata")
    p.add_argument("--stop",   type=str,   choices=["time", "evals"], default="evals",
                   help="Stopping criterion")
    p.add_argument("--time",   type=float, default=1000.0,
                   help="Max wall-clock time in seconds (--stop time)")
    p.add_argument("--evals",  type=int,   default=1650,
                   help="Max function evaluations (--stop evals)")
    return p.parse_args()


# ── Summary printer ───────────────────────────────────────────────────────
def print_summary(all_results):
    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  {'Instance':<22}  {'PH':>8}  {'MH mean':>10}  {'MH best':>8}  {'ARPD':>7}  {'time':>7}")
    print(sep)
    for r in all_results:
        print(f"  {r['label']:<22}  {r['ph_obj']:>8.2f}  "
              f"{r['mh_mean']:>10.4f}  {r['mh_best']:>8.4f}  "
              f"{r['mh_arpd']:>6.2f}%  {r['mh_time']:>6.1f}s")
    print(sep)


# ── Per-instance runner ───────────────────────────────────────────────────
def run_instance(inst, label, args, params_base, stop_label):
    """Run PH + MH on *inst* and return a result dict."""
    print(f"\n{'='*60}\nInstance: {label}\n{'='*60}")
    print(f"  {inst.n_ops} ops | {inst.n_machines} machines | "
          f"{len(inst.tool_sizes)} tool sets | capacity={inst.magazine_capacity}")

    # PH reference (run args.runs times for a stable mean)
    ph_vals = []
    for run in range(args.runs):
        random.seed(run)
        _, v = practitioner_heuristic(inst)
        ph_vals.append(v)
    ph_obj = sum(ph_vals) / args.runs
    ph_std = (sum((v - ph_obj)**2 for v in ph_vals) / args.runs) ** 0.5
    print(f"  PH objective : {ph_obj:.2f} ± {ph_std:.2f}\n")

    # MH runs
    print(f"  [MH] running {args.runs} runs …")
    fitnesses, times, gens, run_details = [], [], [], []
    for run in range(args.runs):
        p = GAParams(**{**params_base.__dict__, "seed": args.seed + run})
        r = matheuristic(inst, p)
        fitnesses.append(r.best_fitness)
        times.append(r.computation_time)
        gens.append(r.n_generations)
        run_details.append((r.best_fitness, r.computation_time, r.n_generations))
        print(f"  Run {run+1:2d}: fitness={r.best_fitness:.4f}  "
              f"time={r.computation_time:.1f}s  gens={r.n_generations}")

    mean = sum(fitnesses) / args.runs
    std  = (sum((f - mean)**2 for f in fitnesses) / args.runs) ** 0.5
    best = min(fitnesses)
    arpd = 100 * sum((f - ph_obj) / max(ph_obj, 1e-9) for f in fitnesses) / args.runs

    print(f"  MH mean={mean:.4f} ± {std:.4f}  best={best:.4f}  "
          f"time={sum(times)/args.runs:.1f}s  ARPD={arpd:.2f}%")

    return {
        "label":          label,
        "n_ops":          inst.n_ops,
        "n_machines":     inst.n_machines,
        "n_tools":        len(inst.tool_sizes),
        "capacity":       inst.magazine_capacity,
        "stop_criterion": stop_label,
        "ph_obj":         ph_obj,
        "ph_std":         ph_std,
        "mh_mean":        mean,
        "mh_std":         std,
        "mh_best":        best,
        "mh_time":        sum(times) / args.runs,
        "mh_gens":        sum(gens)  / args.runs,
        "mh_arpd":        arpd,
        "mh_run_details": run_details,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.output is None:
        if args.case == "table8":
            args.output = str(OUT / "results_mh_table8.xlsx")
        else:
            args.output = str(OUT / "results_mh_base.xlsx")

    # Validate --case
    valid = set(BASE_CASES.keys()) | {"table8", "all"}
    if args.case not in valid:
        print(f"[ERROR] Unknown --case '{args.case}'. Options: {sorted(valid)}")
        sys.exit(1)

    # Build GAParams
    if args.stop == "time":
        stop_label  = f"time={args.time}s"
        params_base = GAParams(np_size=args.pop, max_time=args.time,
                               max_evals=10**9, Gc=args.gc, seed=args.seed)
    else:
        stop_label  = f"evals={args.evals}"
        params_base = GAParams(np_size=args.pop, max_time=10**9,
                               max_evals=args.evals, Gc=args.gc, seed=args.seed)

    print(f"Algorithm : MH (base matheuristic)")
    print(f"Stop      : {stop_label}  |  pop={args.pop}  |  Gc={args.gc}  |  runs={args.runs}")

    all_results = []

    # ── Table 8 mode ──────────────────────────────────────────────────────
    if args.case == "table8":
        csv_path = str(BASE / "6M140" / "6M140.csv")
        if not os.path.exists(csv_path):
            print(f"[ERROR] 6M140 CSV not found: {csv_path}")
            sys.exit(1)
        full_inst = load_instance_csv(csv_path)
        print(f"\nTable-8 mode: {len(TABLE_8)} sub-instances from 6M140 "
              f"({full_inst.n_ops} ops total)\n")
        for n, label in TABLE_8.items():
            inst   = _make_subinstance(full_inst, n)
            result = run_instance(inst, label, args, params_base, stop_label)
            all_results.append(result)

    # ── Normal mode ───────────────────────────────────────────────────────
    else:
        cases_to_run = BASE_CASES if args.case == "all" else {args.case: BASE_CASES[args.case]}
        for key, (csv_path, label) in cases_to_run.items():
            if not os.path.exists(csv_path):
                print(f"[WARNING] Not found: {csv_path} — skipping {key}")
                continue
            inst   = load_instance_csv(csv_path)
            result = run_instance(inst, label, args, params_base, stop_label)
            all_results.append(result)

    # ── Output ────────────────────────────────────────────────────────────
    if all_results:
        print_summary(all_results)
        meta = {
            "run_name":  args.name if args.name else args.output,
            "timestamp": datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
            "params": {
                "stop":                        args.stop,
                stop_label.split("=")[0]:      stop_label.split("=")[1],
                "pop":                         args.pop,
                "gc":                          args.gc,
                "runs":                        args.runs,
                "seed":                        args.seed,
                "case":                        args.case,
            },
        }
        save_mh_only_excel(all_results, args.output, meta=meta)
    else:
        print(f"\n[ERROR] No cases found. Looking in: {BASE}")


if __name__ == "__main__":
    main()
