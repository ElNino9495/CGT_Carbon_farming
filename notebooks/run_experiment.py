#!/usr/bin/env python3
"""
run_experiment.py
Run the full 4-notebook pipeline for one experiment config.
Usage: python run_experiment.py --name CF1_baseline --cf_rule CF1 --ccp 1500 --delta_mrv 0.7
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

def make_config_overrides(args) -> dict:
    return {
        "EXPERIMENT_NAME" : args.name,
        "CF_RULE"         : args.cf_rule,
        "CCP"             : args.ccp,
        "DELTA_MRV"       : args.delta_mrv,
        "VARIABLE_MRV"    : args.variable_mrv,
        "FIXED_MRV"       : args.fixed_mrv,
        "BUDGET_MULTIPLIER": args.budget_multiplier,
        "PADDY_PRICE" : args.paddy_price,
        "FIXED_T"     : args.fixed_t,
        "VARIABLE_T"  : args.variable_t,
        "N_JOBS"          : args.n_jobs,  # ← important for parallel runs
    }

def write_config_patch(overrides: dict, patch_path: Path):
    """Write a small Python file that patches config values after import."""
    lines = [
        "# Auto-generated config patch — do not edit manually\n",
        "import sys, importlib\n",
        "sys.path.insert(0, str(__file__))\n",
    ]
    for k, v in overrides.items():
        if isinstance(v, str):
            lines.append(f'{k} = "{v}"\n')
        else:
            lines.append(f"{k} = {v}\n")
    patch_path.write_text("".join(lines))

def run_notebook_as_script(script_path: Path, env: dict) -> bool:
    """Run a converted notebook script, return True if successful."""
    result = subprocess.run(
        [sys.executable, str(script_path)],
        env={**os.environ, **env},
        capture_output=False,  # stream output live
    )
    return result.returncode == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",             required=True)
    parser.add_argument("--cf_rule",          default="CF1")
    parser.add_argument("--ccp",              type=float, default=1500)
    parser.add_argument("--delta_mrv",        type=float, default=0.7)
    parser.add_argument("--variable_mrv",     type=float, default=2000)
    parser.add_argument("--fixed_mrv",        type=float, default=5000)
    parser.add_argument("--budget_multiplier",type=float, default=1.0)
    parser.add_argument("--paddy_price",  type=float, default=22000)
    parser.add_argument("--fixed_t",      type=float, default=2000)
    parser.add_argument("--variable_t",   type=float, default=500)
    parser.add_argument("--n_jobs",           type=int,   default=4)  # NOT -1 when parallel
    args = parser.parse_args()

    # Pass overrides via environment variables — read in config.py
    env_overrides = {f"EXP_{k}": str(v) for k, v in make_config_overrides(args).items()}

    notebooks_dir = Path(__file__).resolve().parent
    scripts = [
        notebooks_dir / "notebook_01_data_ingestion.py",
        notebooks_dir / "notebook_02_solo_optimisation.py",
        notebooks_dir / "notebook_03_coalition_enumeration.py",
        notebooks_dir / "notebook_04_surplus_allocation.py",
    ]

    for script in scripts:
        print(f"\n[{args.name}] Running {script.name}...")
        ok = run_notebook_as_script(script, env_overrides)
        if not ok:
            print(f"[{args.name}] FAILED at {script.name}. Aborting.")
            sys.exit(1)

    print(f"\n[{args.name}] Complete.")