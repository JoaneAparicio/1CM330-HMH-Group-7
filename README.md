# Hybrid Meta-Heuristics for Parallel Machine Scheduling with Tool Replacements

Implementation of a Genetic Algorithm Matheuristic (MH) and a Memetic Algorithm (MH + Local Search) for the **Parallel Machine Scheduling with Tool Replacements** problem, based on Dang et al. (2021).

## Problem

Each job consists of operations that require a specific tool set. Machines have a limited tool magazine (capacity C). When a required tool is not loaded on a machine, a tool replacement must be performed at cost τ. The objective minimises a weighted sum of **tardiness** and **tool setup costs** across all operations.

A solution is encoded as a chromosome `(VI, VM)`:
- `VI` — job-order vector (sequence gene)
- `VM` — machine-assignment vector (assignment gene)

## Algorithms

### Practitioner Heuristic (PH)
Two-phase reference heuristic (Algorithm 4, Dang et al. 2021): initial tool allocation followed by EDD-ordered operation assignment. Used to seed the population and as a baseline for ARPD computation.

### Base Matheuristic (MH)
Genetic algorithm with:
- **Selection**: tournament selection (γ₁ = 0.20)
- **Crossover**: Problem-Oriented Crossover (POX) or Combined Crossover (CX), switched by a phase flag (`B = 1`)
- **Mutation**: swap mutation on VI + uniform mutation on VM (Ps = Pu = 0.01)
- **Survivor selection**: elitism (γ₂ = 0.10) + immigration (duplicates replaced by random chromosomes)
- **Fitness**: Tool Replacement Method (TRM) simulation with optional ILP eviction (PuLP), falling back to greedy

### Memetic Algorithm (MH + LS)
Extends the base MH with a dedicated **Local Search branch** that runs each generation alongside the crossover/mutation pipeline:
- `LS_SHARE = 1/3` of offspring are produced by local search descent
- LS operators: intra-machine resequencing, inter-machine relocate, inter-machine swap (VM-only variants also available)
- ILS loop: descent until local optimum, perturb VM, repeat

## Project Structure

```
.
├── models.py                   # Data structures and chromosome encoding
├── evaluation.py               # TRM fitness evaluation (ILP + greedy fallback)
├── ga_operators.py             # Crossover, mutation, selection, PH, MH loop
├── local_search.py             # Intra/inter-machine LS operators, ILS, VM-only variants
├── data_loader.py              # CSV instance loader
├── export.py                   # Excel export (results + metadata)
├── run_mh.py                   # CLI runner for the base Matheuristic
├── run_mh_memetic_final.py     # Runner for the Memetic Algorithm
├── compare_stats.py            # Box plots + Wilcoxon / paired t-test comparison
│
├── data/
│   ├── 2M38/                   # 2-machine, 38-operation base instance + scenarios
│   ├── 2M46/                   # 2-machine, 46-operation base instance + scenarios
│   ├── 6M140/                  # 6-machine, 140-operation base instance + scenarios
│   ├── 6M163/                  # 6-machine, 163-operation base instance + scenarios
│   └── ParameterTuning/        # Instances used for parameter tuning
│
├── table8_instances/           # Auto-generated sub-instances of 6M140 (n = 15…140)
└── output/                     # Excel results and comparison plots
```

Each instance folder contains three sensitivity-analysis scenarios:
- **Scenario 1** — varying tool magazine ratio R
- **Scenario 2** — varying tool cost C
- **Scenario 3** — varying due-date tightness D

## Usage

### Run the base Matheuristic

```bash
# Default: Table 8 sub-instances of 6M140, 1650 evaluations, 10 runs
python run_mh.py

# Paper replication (time budget)
python run_mh.py --runs 10 --stop time --time 3600

# Single base instance, quick test
python run_mh.py --case 2M38 --runs 3 --stop evals --evals 500

# All base cases
python run_mh.py --case all
```

**Arguments**

| Flag | Default | Description |
|------|---------|-------------|
| `--case` | `table8` | `table8` \| `2M38` \| `2M46` \| `6M140` \| `6M163` \| `all` |
| `--runs` | `10` | Runs per instance |
| `--pop` | `100` | Population size (Np) |
| `--gc` | `100` | No-improvement generations before stopping (Gc) |
| `--stop` | `evals` | Stopping criterion: `evals` or `time` |
| `--evals` | `1650` | Max function evaluations |
| `--time` | `1000` | Max wall-clock seconds (when `--stop time`) |
| `--seed` | `42` | Base random seed |
| `--output` | auto | Output Excel file path |

### Run the Memetic Algorithm

Edit the `CONFIGURATION` block at the top of `run_mh_memetic_final.py`, then:

```bash
python run_mh_memetic_final.py
```

Key configuration parameters:

```python
NFC_BUDGET      = 1650      # evaluation budget
RUNS            = 10        # runs per instance
POPULATION_SIZE = 100       # Np
LS_SHARE        = 1/3       # fraction of offspring from the LS branch
GC              = 100       # no-improvement generations to stop
```

### Compare MH vs Memetic

After running both algorithms, place the Excel outputs in `output/` and run:

```bash
python compare_stats.py
```

This produces:
- `output/compare_base_cases.png` — box plots for the 4 base instances
- `output/compare_table8.png` — box plots for the 7 Table-8 sub-instances
- `output/statistical_tests.xlsx` — Wilcoxon signed-rank + paired t-test results with ARPD

## Dependencies

```
numpy
scipy
matplotlib
openpyxl
pulp          # optional – enables ILP-based TRM; falls back to greedy if absent
```

Install with:

```bash
pip install numpy scipy matplotlib openpyxl pulp
```

## Output Format

Results are saved as Excel workbooks with:
- **Summary sheet** — mean, std, best, ARPD per instance, PH reference
- **Per-instance detail sheets** — individual run fitness, time, and generation count

ARPD is computed against the Practitioner Heuristic objective:

```
ARPD = mean((f_i - PH) / PH * 100)  over all runs
```

## Reference

Dang, Q. V., Nguyen, C. T., & Salhi, S. (2021). *A Genetic Algorithm for the Parallel Machine Scheduling Problem with Tool Replacements.* (Table 8 sub-instance structure and algorithm design follow this work.)