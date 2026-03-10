import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

@dataclass
class FarmerData:
    farmer_id: str
    farm_size_ha: float
    budget_per_ha: float
    budget_total: float

@dataclass
class PracticeData:
    practice_id: int
    name: str
    base_CSP: float       # tCO2e/ha/season
    base_OC: float        # INR/ha/season
    base_dY: float        # ton/ha/season
    subgroup: str

@dataclass
class DataBundle:
    farmers: List[FarmerData]
    practices: List[PracticeData]
    # farmer_id -> {practice_id: (CSP_ij, y_ij)}
    farmer_practice_effects: Dict[str, Dict[int, Tuple[float, float]]]
    # (i, j) -> alpha_ij (sequestration synergy)
    alpha: Dict[Tuple[int, int], float]
    # (i, j) -> gamma_ij (cost interaction)
    gamma: Dict[Tuple[int, int], float]
    # (i, j) -> delta_ij (yield interaction)
    delta: Dict[Tuple[int, int], float]
    # Set of incompatible practice pairs
    incompatible: Set[Tuple[int, int]]
    # Subgroup mapping: practice_id -> subgroup_id
    subgroup_map: Dict[int, int]
    # Subgroup names
    subgroup_names: List[str]


def _pair_key(i: int, j: int) -> Tuple[int, int]:
    return (min(i, j), max(i, j))


def load_data(data_dir: str = "data/synthetic") -> DataBundle:
    data_path = Path(data_dir)

    # --- Load farmers ---
    df_farmers = pd.read_csv(data_path / "farmers.csv")
    farmers = []
    for _, row in df_farmers.iterrows():
        farmers.append(FarmerData(
            farmer_id=row["Farmer_ID"],
            farm_size_ha=row["Farm_Size_ha"],
            budget_per_ha=row["Budget_per_ha_INR_per_season"],
            budget_total=row["Budget_total_INR_per_season"],
        ))

    # --- Load practices ---
    df_practices = pd.read_csv(data_path / "practices.csv")
    practices = []
    for _, row in df_practices.iterrows():
        practices.append(PracticeData(
            practice_id=int(row["practice_id"]),
            name=row.get("practice_name", f"P{int(row['practice_id'])}"),
            base_CSP=float(row.get("CSP_tCO2e_ha_season", 0)),
            base_OC=float(row.get("OC_INR_ha_season", 0)),
            base_dY=float(row.get("dY_ton_ha_season", 0)),
            subgroup=str(row.get("subgroup", "general")),
        ))

    # --- Load farmer-specific practice effects ---
    df_effects = pd.read_csv(data_path / "farmer_practice_effects.csv")
    farmer_practice_effects: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for _, row in df_effects.iterrows():
        fid = row["farmer_id"]
        pid = int(row["practice_id"])
        csp = float(row["CSP_ij_tCO2e_per_ha_season"])
        y = float(row["y_ij_ton_per_ha_season"])
        if fid not in farmer_practice_effects:
            farmer_practice_effects[fid] = {}
        farmer_practice_effects[fid][pid] = (csp, y)

    # --- Load pairwise interactions ---
    alpha, gamma, delta = {}, {}, {}
    try:
        df_inter = pd.read_csv(data_path / "practice_interactions.csv")
        # Handle unnamed index column: columns are (index=i), j, alpha, delta, gamma
        if "i" not in df_inter.columns:
            df_inter = df_inter.rename(columns={df_inter.columns[0]: "i"})
        for _, row in df_inter.iterrows():
            pk = _pair_key(int(row["i"]), int(row["j"]))
            if "alpha" in row and pd.notna(row["alpha"]):
                alpha[pk] = float(row["alpha"])
            if "gamma" in row and pd.notna(row["gamma"]):
                gamma[pk] = float(row["gamma"])
            if "delta" in row and pd.notna(row["delta"]):
                delta[pk] = float(row["delta"])
    except FileNotFoundError:
        print("Warning: practice_interactions.csv not found, using empty interactions")

    # --- Load incompatibilities ---
    incompatible: Set[Tuple[int, int]] = set()
    try:
        df_incompat = pd.read_csv(data_path / "practice_incompatibility.csv")
        for _, row in df_incompat.iterrows():
            incompatible.add(_pair_key(int(row["practice_id_1"]), int(row["practice_id_2"])))
    except FileNotFoundError:
        print("Warning: practice_incompatibility.csv not found")

    # --- Build subgroup map ---
    subgroup_names_unique = sorted(set(p.subgroup for p in practices))
    sg_name_to_id = {name: idx for idx, name in enumerate(subgroup_names_unique)}
    subgroup_map = {p.practice_id: sg_name_to_id[p.subgroup] for p in practices}

    return DataBundle(
        farmers=farmers,
        practices=practices,
        farmer_practice_effects=farmer_practice_effects,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        incompatible=incompatible,
        subgroup_map=subgroup_map,
        subgroup_names=subgroup_names_unique,
    )