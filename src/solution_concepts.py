"""
Cooperative game theory solution concepts:
  - Shapley value
  - Core membership check
  - Nucleolus (via LP)
"""
import math
from typing import Dict, FrozenSet, List
from itertools import combinations

from .coalition import CoalitionGame


def shapley_value(
    game: CoalitionGame,
    grand_coalition: FrozenSet[str],
) -> Dict[str, float]:
    """
    Compute exact Shapley values for the grand coalition.

    Warning: O(2^n) — feasible for n <= ~20-22 farmers.
    For larger n, use sampling-based approximation.
    """
    players = list(grand_coalition)
    n = len(players)
    phi = {p: 0.0 for p in players}

    # Pre-compute all coalition values
    value_cache: Dict[FrozenSet[str], float] = {}

    def v(S: FrozenSet[str]) -> float:
        if S not in value_cache:
            value_cache[S] = game.characteristic_value(S)
        return value_cache[S]

    for i, player in enumerate(players):
        others = [p for p in players if p != player]
        for k in range(0, n):
            # Subsets of others of size k
            for combo in combinations(others, k):
                S = frozenset(combo)
                S_with_i = S | {player}
                marginal = v(S_with_i) - v(S)
                weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                phi[player] += weight * marginal

    return phi


def shapley_value_sampling(
    game: CoalitionGame,
    grand_coalition: FrozenSet[str],
    n_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Approximate Shapley values via random permutation sampling.
    Suitable for n > 20.
    """
    import random
    rng = random.Random(seed)
    players = list(grand_coalition)
    n = len(players)
    phi = {p: 0.0 for p in players}
    value_cache: Dict[FrozenSet[str], float] = {}

    def v(S: FrozenSet[str]) -> float:
        if S not in value_cache:
            value_cache[S] = game.characteristic_value(S)
        return value_cache[S]

    for _ in range(n_samples):
        perm = players[:]
        rng.shuffle(perm)
        S = frozenset()
        for player in perm:
            S_with = S | {player}
            marginal = v(S_with) - v(S)
            phi[player] += marginal
            S = S_with

    for p in phi:
        phi[p] /= n_samples

    return phi


def is_in_core(
    game: CoalitionGame,
    allocation: Dict[str, float],
    grand_coalition: FrozenSet[str],
) -> bool:
    """
    Check if an allocation is in the core.
    An allocation x is in the core if for every sub-coalition S:
      sum_{i in S} x_i >= v(S)
    and sum_{i in N} x_i = v(N).
    """
    players = list(grand_coalition)
    n = len(players)

    # Check efficiency
    total_alloc = sum(allocation[p] for p in players)
    v_grand = game.characteristic_value(grand_coalition)
    if abs(total_alloc - v_grand) > 1e-6:
        return False

    # Check all sub-coalitions
    for k in range(1, n):
        for combo in combinations(players, k):
            S = frozenset(combo)
            v_S = game.characteristic_value(S)
            alloc_S = sum(allocation[p] for p in S)
            if alloc_S < v_S - 1e-6:
                return False

    return True


def nucleolus(
    game: CoalitionGame,
    grand_coalition: FrozenSet[str],
) -> Dict[str, float]:
    """
    Compute the nucleolus via iterated LP (Maschler et al.).
    Uses Gurobi if available, else PuLP.
    """
    try:
        return _nucleolus_gurobi(game, grand_coalition)
    except ImportError:
        return _nucleolus_pulp(game, grand_coalition)


def _nucleolus_gurobi(
    game: CoalitionGame,
    grand_coalition: FrozenSet[str],
) -> Dict[str, float]:
    import gurobipy as gp
    from gurobipy import GRB

    players = list(grand_coalition)
    n = len(players)
    v_grand = game.characteristic_value(grand_coalition)

    # Enumerate all proper sub-coalitions
    sub_coalitions = []
    for k in range(1, n):
        for combo in combinations(players, k):
            sub_coalitions.append(frozenset(combo))

    v_sub = {S: game.characteristic_value(S) for S in sub_coalitions}

    model = gp.Model("nucleolus")
    model.setParam("OutputFlag", 0)

    x = {p: model.addVar(lb=-GRB.INFINITY, name=f"x_{p}") for p in players}
    epsilon = model.addVar(lb=-GRB.INFINITY, name="epsilon")

    model.setObjective(epsilon, GRB.MAXIMIZE)

    # Efficiency: sum x_i = v(N)
    model.addConstr(gp.quicksum(x[p] for p in players) == v_grand)

    # For each sub-coalition S: sum_{i in S} x_i >= v(S) + epsilon
    for S in sub_coalitions:
        model.addConstr(
            gp.quicksum(x[p] for p in S) >= v_sub[S] + epsilon,
            name=f"excess_{hash(S)}"
        )

    model.optimize()

    return {p: x[p].X for p in players}


def _nucleolus_pulp(
    game: CoalitionGame,
    grand_coalition: FrozenSet[str],
) -> Dict[str, float]:
    import pulp

    players = list(grand_coalition)
    n = len(players)
    v_grand = game.characteristic_value(grand_coalition)

    sub_coalitions = []
    for k in range(1, n):
        for combo in combinations(players, k):
            sub_coalitions.append(frozenset(combo))

    v_sub = {S: game.characteristic_value(S) for S in sub_coalitions}

    prob = pulp.LpProblem("nucleolus", pulp.LpMaximize)

    x = {p: pulp.LpVariable(f"x_{p}") for p in players}
    epsilon = pulp.LpVariable("epsilon")

    prob += epsilon

    prob += pulp.lpSum(x[p] for p in players) == v_grand

    for S in sub_coalitions:
        prob += pulp.lpSum(x[p] for p in S) >= v_sub[S] + epsilon

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return {p: x[p].varValue for p in players}