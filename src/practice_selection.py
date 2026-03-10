"""
Optimal Practice Selection for individual farmers (Section 3.1).

Solves:
  max  CSP(Pi) * CCP + Y(Pi) * CROP_PRICE - OC(Pi)
  s.t. OC(Pi) <= B
       incompatibility constraints
       at most one practice per subgroup
"""
from typing import Dict, List, Set, Tuple, Optional
from itertools import combinations
import numpy as np

from .config import ModelConfig
from .data_loader import DataBundle, FarmerData, _pair_key


def _compute_farmer_coefficients(
    farmer: FarmerData,
    data: DataBundle,
    config: ModelConfig,
) -> Tuple[List[float], Dict[Tuple[int, int], float], List[float], Dict[Tuple[int, int], float]]:
    """
    Compute linear and pairwise objective terms for a farmer.

    Returns:
        b: linear objective coefficient for each practice
        w: pairwise objective coefficient for each interacting pair
        oc: operational cost per practice
        gc: pairwise cost interactions
    """
    fid = farmer.farmer_id
    effects = data.farmer_practice_effects.get(fid, {})
    practice_ids = [p.practice_id for p in data.practices]
    m = len(practice_ids)

    b = []
    oc = []
    for p in data.practices:
        pid = p.practice_id
        if pid in effects:
            csp_ij, y_ij = effects[pid]
        else:
            csp_ij, y_ij = p.base_CSP, p.base_dY

        revenue = config.CCP * csp_ij + config.CROP_PRICE * y_ij
        cost = p.base_OC
        b.append(revenue - cost)
        oc.append(cost)

    # Pairwise terms
    w = {}
    gc = {}
    pid_to_idx = {p.practice_id: i for i, p in enumerate(data.practices)}

    all_pairs = set(data.alpha.keys()) | set(data.delta.keys()) | set(data.gamma.keys())
    for (pi, pj) in all_pairs:
        if pi not in pid_to_idx or pj not in pid_to_idx:
            continue
        i, j = pid_to_idx[pi], pid_to_idx[pj]
        key = _pair_key(i, j)

        alpha_val = data.alpha.get(_pair_key(pi, pj), 0.0)
        delta_val = data.delta.get(_pair_key(pi, pj), 0.0)
        gamma_val = data.gamma.get(_pair_key(pi, pj), 0.0)

        w_val = config.CCP * alpha_val + config.CROP_PRICE * delta_val - gamma_val
        if abs(w_val) > 1e-12:
            w[key] = w_val
        if abs(gamma_val) > 1e-12:
            gc[key] = gamma_val

    return b, w, oc, gc


def solve_practice_selection_gurobi(
    farmer: FarmerData,
    data: DataBundle,
    config: ModelConfig,
) -> Tuple[Set[int], float, float]:
    """
    Solve the optimal practice selection using Gurobi MIQP.

    Returns:
        selected: set of selected practice indices (0-based)
        obj_value: optimal objective value (INR/ha/season)
        total_cost: total operational cost
    """
    # import gurobipy as gp
    # from gurobipy import GRB

    b, w, oc, gc = _compute_farmer_coefficients(farmer, data, config)
    m = len(data.practices)
    budget = farmer.budget_per_ha

    pid_to_idx = {p.practice_id: i for i, p in enumerate(data.practices)}

    model = gp.Model(f"practice_selection_{farmer.farmer_id}")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", config.time_limit)
    model.setParam("MIPGap", config.mip_gap)

    # Decision variables
    x = model.addVars(m, vtype=GRB.BINARY, name="x")

    # Objective: maximize linear + quadratic terms
    obj = gp.QuadExpr()
    for j in range(m):
        obj += b[j] * x[j]
    for (i, j), w_val in w.items():
        obj += w_val * x[i] * x[j]

    model.setObjective(obj, GRB.MAXIMIZE)

    # Budget constraint: OC(Pi) <= B
    cost_expr = gp.QuadExpr()
    for j in range(m):
        cost_expr += oc[j] * x[j]
    for (i, j), g_val in gc.items():
        cost_expr += g_val * x[i] * x[j]

    model.addQConstr(cost_expr <= budget, name="budget")

    # Incompatibility constraints
    for (pi, pj) in data.incompatible:
        if pi in pid_to_idx and pj in pid_to_idx:
            i, j = pid_to_idx[pi], pid_to_idx[pj]
            model.addConstr(x[i] + x[j] <= 1, name=f"incompat_{pi}_{pj}")

    # Subgroup constraints: at most one per subgroup
    subgroups: Dict[int, List[int]] = {}
    for i, p in enumerate(data.practices):
        sg = data.subgroup_map[p.practice_id]
        subgroups.setdefault(sg, []).append(i)
    for sg_id, members in subgroups.items():
        if len(members) > 1:
            model.addConstr(
                gp.quicksum(x[i] for i in members) <= 1,
                name=f"subgroup_{sg_id}"
            )

    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        selected = {j for j in range(m) if x[j].X > 0.5}
        return selected, model.ObjVal, sum(oc[j] for j in selected)
    else:
        return set(), 0.0, 0.0


def solve_practice_selection_pulp(
    farmer: FarmerData,
    data: DataBundle,
    config: ModelConfig,
) -> Tuple[Set[int], float, float]:
    """
    Fallback: linearized MILP via PuLP (for users without Gurobi).
    Introduces auxiliary y_ij = x_i * x_j variables.
    """
    import pulp

    b, w, oc, gc = _compute_farmer_coefficients(farmer, data, config)
    m = len(data.practices)
    budget = farmer.budget_per_ha
    pid_to_idx = {p.practice_id: i for i, p in enumerate(data.practices)}

    prob = pulp.LpProblem(f"practice_{farmer.farmer_id}", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(m)]

    # Auxiliary variables for quadratic terms
    y_vars = {}
    for (i, j) in set(w.keys()) | set(gc.keys()):
        y = pulp.LpVariable(f"y_{i}_{j}", cat="Binary")
        y_vars[(i, j)] = y
        # McCormick / linearization: y <= x_i, y <= x_j, y >= x_i + x_j - 1
        prob += y <= x[i]
        prob += y <= x[j]
        prob += y >= x[i] + x[j] - 1

    # Objective
    obj = pulp.lpSum(b[j] * x[j] for j in range(m))
    obj += pulp.lpSum(w_val * y_vars[key] for key, w_val in w.items() if key in y_vars)
    prob += obj

    # Budget
    cost_expr = pulp.lpSum(oc[j] * x[j] for j in range(m))
    cost_expr += pulp.lpSum(g_val * y_vars[key] for key, g_val in gc.items() if key in y_vars)
    prob += cost_expr <= budget

    # Incompatibility
    for (pi, pj) in data.incompatible:
        if pi in pid_to_idx and pj in pid_to_idx:
            i, j = pid_to_idx[pi], pid_to_idx[pj]
            prob += x[i] + x[j] <= 1

    # Subgroup constraints
    subgroups: Dict[int, List[int]] = {}
    for i, p in enumerate(data.practices):
        sg = data.subgroup_map[p.practice_id]
        subgroups.setdefault(sg, []).append(i)
    for sg_id, members in subgroups.items():
        if len(members) > 1:
            prob += pulp.lpSum(x[i] for i in members) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=config.time_limit))

    if prob.status == pulp.constants.LpStatusOptimal:
        selected = {j for j in range(m) if x[j].varValue > 0.5}
        return selected, pulp.value(prob.objective), sum(oc[j] for j in selected)
    else:
        return set(), 0.0, 0.0


def solve_all_farmers(
    data: DataBundle,
    config: ModelConfig,
) -> Dict[str, Tuple[Set[int], float, float]]:
    """
    Solve optimal practice selection for all farmers.

    Returns:
        dict mapping farmer_id -> (selected_practices, obj_value, total_cost)
    """
    solver_fn = (
        solve_practice_selection_gurobi if config.solver == "gurobi"
        else solve_practice_selection_pulp
    )

    results = {}
    for farmer in data.farmers:
        selected, obj, cost = solver_fn(farmer, data, config)
        results[farmer.farmer_id] = (selected, obj, cost)
        practice_names = [data.practices[j].name for j in sorted(selected)]
        print(f"  {farmer.farmer_id}: {len(selected)} practices, "
              f"obj={obj:.2f}, cost={cost:.2f} — {practice_names}")

    return results