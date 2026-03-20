# launch_experiments.py
import subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# =============================================================================
#  EVERY KNOB YOU CAN TURN
# =============================================================================
#
#  name              — unique experiment folder name (string)
#  cf_rule           — "CF1" | "CF2" | "CF3"
#                        CF1: pooled residual (most permissive, always linear)
#                        CF2: area-weighted per-farmer share (linear, stricter)
#                        CF3: sequestration-weighted share (bilinear, strictest)
#
#  ccp               — carbon credit price (INR / tCO2e)
#                        base: 1500   range tested: 1000–5000
#
#  delta_mrv         — MRV scale exponent on total coalition area
#                        0.0  → flat MRV (no economies of scale)
#                        0.6  → moderate economies of scale
#                        0.7  → base value
#                        1.0  → fully linear (no scale benefit)
#
#  fixed_mrv         — flat MRV cost per certification event (INR)
#                        base: 5000   lower → easier solo participation
#
#  variable_mrv      — variable MRV rate (INR / ha^delta_mrv)
#                        base: 2000   higher → stronger coalition incentive
#
#  fixed_t           — flat transaction / registry fee per certification (INR)
#                        base: 2000
#
#  variable_t        — per-member transaction cost (INR / farmer)
#                        base: 500    higher → penalises large coalitions
#
#  budget_multiplier — scales every farmer's Budget_total_INR_per_season
#                        1.0 → base budgets from CSV
#                        2.0 → double budgets (farmers can afford more practices)
#                        0.5 → half budgets (tighter feasibility)
#
#  paddy_price       — farm-gate paddy price (INR / ton)
#                        base: 22000  affects yield revenue in objective
#
#  n_jobs            — joblib workers inside notebook_03 coalition enumeration
#                        set < total_cores / max_workers to avoid CPU contention
#                        e.g. 16-core machine, 3 parallel experiments → n_jobs=5
#
# =============================================================================

experiments = [
    # (name, cf_rule, ccp, delta_mrv, fixed_mrv, variable_mrv, fixed_t, variable_t, budget_mult, paddy_price, n_jobs)
    
    # ── CF1 × CCP 1500–3000 (Fixed: Delta 0.7, MRV 5000/2000, T 2000/500) ────────────────
    ("CF1_ccp1500", "CF1", 1500, 0.7, 5000, 2000, 2000, 500, 1.0, 22000, 4),
    ("CF1_ccp2000", "CF1", 2000, 0.7, 5000, 2000, 2000, 500, 1.0, 22000, 4),
    ("CF1_ccp2500", "CF1", 2500, 0.7, 5000, 2000, 2000, 500, 1.0, 22000, 4),
    ("CF1_ccp3000", "CF1", 3000, 0.7, 5000, 2000, 2000, 500, 1.0, 22000, 4),

    # ── CF2 × CCP 1500–3000 (Fixed: Delta 0.7, MRV 5000/2000, T 2000/500) ────────────────
    ("CF2_ccp1500", "CF2", 1500, 0.7, 5000, 2000, 2000, 500, 1.0, 22000, 4),
    ("CF2_ccp2000", "CF2", 2000, 0.7, 5000, 2000, 2000, 500, 1.0, 22000, 4),
    ("CF2_ccp2500", "CF2", 2500, 0.7, 5000, 2000, 2000, 500, 1.0, 22000, 4),
    ("CF2_ccp3000", "CF2", 3000, 0.7, 5000, 2000, 2000, 500, 1.0, 22000, 4),
]

def run_one(exp):
    (name, cf_rule, ccp, delta_mrv,
     fixed_mrv, variable_mrv, fixed_t, variable_t,
     budget_multiplier, paddy_price, n_jobs) = exp

    from pathlib import Path
    script_path = Path(__file__).resolve().parent / "run_experiment.py"

    result = subprocess.run([
        sys.executable, str(script_path),
        "--name",              name,
        "--cf_rule",           cf_rule,
        "--ccp",               str(ccp),
        "--delta_mrv",         str(delta_mrv),
        "--fixed_mrv",         str(fixed_mrv),
        "--variable_mrv",      str(variable_mrv),
        "--fixed_t",           str(fixed_t),
        "--variable_t",        str(variable_t),
        "--budget_multiplier", str(budget_multiplier),
        "--paddy_price",       str(paddy_price),
        "--n_jobs",            str(n_jobs),
    ])
    return name, result.returncode == 0


if __name__ == "__main__":
    MAX_PARALLEL = 10   # tune to: floor(total_cores / n_jobs)

    print(f"Launching {len(experiments)} experiments, {MAX_PARALLEL} at a time...\n")

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(run_one, exp): exp[0] for exp in experiments}
        for f in as_completed(futures):
            name, ok = f.result()
            print(f"{'✓ OK  ' if ok else '✗ FAIL'}: {name}")