"""
Coalition formation with heterogeneous practice portfolios (Section 3.2).

Implements:
  - Characteristic function v(S, Pi_S)
  - Singleton value v_tilde({i}, Pi_i*)
  - Individual rationality checks
  - Homogeneous vs heterogeneous comparison
"""
from typing import Dict, FrozenSet, List, Set, Tuple
from itertools import combinations
import numpy as np

from .config import ModelConfig
from .data_loader import DataBundle, FarmerData, _pair_key


class CoalitionGame:
    def __init__(
        self,
        data: DataBundle,
        config: ModelConfig,
        farmer_portfolios: Dict[str, Set[int]],  # farmer_id -> set of practice indices
    ):
        self.data = data
        self.config = config
        self.portfolios = farmer_portfolios

        # Pre-compute farmer-level sequestration and cost
        self._farmer_csp: Dict[str, float] = {}
        self._farmer_oc: Dict[str, float] = {}
        self._farmer_yield: Dict[str, float] = {}

        for farmer in data.farmers:
            fid = farmer.farmer_id
            portfolio = farmer_portfolios.get(fid, set())
            csp, oc, dy = self._compute_farmer_values(farmer, portfolio)
            self._farmer_csp[fid] = csp
            self._farmer_oc[fid] = oc
            self._farmer_yield[fid] = dy

    def _compute_farmer_values(
        self, farmer: FarmerData, portfolio: Set[int]
    ) -> Tuple[float, float, float]:
        """Compute CSP_i(Pi_i*), OC_i(Pi_i*), Y_i(Pi_i*) for a farmer."""
        effects = self.data.farmer_practice_effects.get(farmer.farmer_id, {})
        practices = self.data.practices
        pid_to_idx = {p.practice_id: i for i, p in enumerate(practices)}

        # Linear terms
        csp = 0.0
        oc = 0.0
        dy = 0.0
        practice_ids_in_portfolio = []

        for j in portfolio:
            p = practices[j]
            pid = p.practice_id
            practice_ids_in_portfolio.append(pid)

            if pid in effects:
                csp_ij, y_ij = effects[pid]
            else:
                csp_ij, y_ij = p.base_CSP, p.base_dY

            csp += csp_ij
            oc += p.base_OC
            dy += y_ij

        # Pairwise interaction terms
        for j, k in combinations(sorted(portfolio), 2):
            pj = practices[j].practice_id
            pk = practices[k].practice_id
            pair = _pair_key(pj, pk)

            csp += self.data.alpha.get(pair, 0.0)
            oc += self.data.gamma.get(pair, 0.0)
            dy += self.data.delta.get(pair, 0.0)

        return csp, oc, dy

    def _certification_cost(
        self,
        coalition_farmer_ids: List[str],
        union_practices: Set[int],
    ) -> float:
        """
        Compute total certification cost: C_MR + C_V + C_T (Equations 7-9).
        """
        cfg = self.config
        total_area = sum(
            f.farm_size_ha for f in self.data.farmers
            if f.farmer_id in coalition_farmer_ids
        )
        n_farmers = len(coalition_farmer_ids)
        n_distinct = len(union_practices)

        c_mr = (cfg.fixed_MR
                + cfg.variable_MR * (total_area ** cfg.alpha_MR)
                + cfg.phi_MR * n_distinct)

        c_v = (cfg.fixed_V
               + cfg.variable_V * (total_area ** cfg.beta_V)
               + cfg.phi_V * n_distinct)

        c_t = cfg.fixed_T + cfg.variable_T * n_farmers

        return c_mr + c_v + c_t

    def characteristic_value(self, coalition: FrozenSet[str]) -> float:
        """
        Compute v(S, Pi_S) — Equation 10.
        Coalition-level carbon credit revenue minus shared certification costs.
        OC is excluded (private cost handled via IR).
        """
        farmer_ids = list(coalition)
        if not farmer_ids:
            return 0.0

        # Aggregate sequestration weighted by farm size
        total_csp = 0.0
        union_practices: Set[int] = set()

        for fid in farmer_ids:
            farmer = next(f for f in self.data.farmers if f.farmer_id == fid)
            portfolio = self.portfolios.get(fid, set())
            total_csp += farmer.farm_size_ha * self._farmer_csp[fid]
            union_practices.update(portfolio)

        revenue = total_csp * self.config.CCP
        cert_cost = self._certification_cost(farmer_ids, union_practices)

        return revenue - cert_cost

    def singleton_value(self, farmer_id: str) -> float:
        """
        Compute v_tilde({i}, Pi_i*) — Equation 12.
        Includes carbon revenue, yield revenue, OC, and solo certification costs.
        """
        farmer = next(f for f in self.data.farmers if f.farmer_id == farmer_id)
        portfolio = self.portfolios.get(farmer_id, set())

        csp = self._farmer_csp[farmer_id]
        oc = self._farmer_oc[farmer_id]
        dy = self._farmer_yield[farmer_id]

        revenue = (farmer.farm_size_ha * csp * self.config.CCP
                   + farmer.farm_size_ha * dy * self.config.CROP_PRICE)
        cost = farmer.farm_size_ha * oc
        cert_cost = self._certification_cost([farmer_id], portfolio)

        return revenue - cost - cert_cost

    def coalition_surplus(self, coalition: FrozenSet[str]) -> float:
        """
        Surplus from coalition formation = v(S) - sum of singleton values.
        """
        v_coalition = self.characteristic_value(coalition)
        v_singletons = sum(self.singleton_value(fid) for fid in coalition)
        return v_coalition - v_singletons

    def is_individually_rational(
        self, allocation: Dict[str, float], coalition: FrozenSet[str]
    ) -> bool:
        """Check if allocation satisfies IR for all members (Equation 13)."""
        for fid in coalition:
            if allocation.get(fid, 0.0) < self.singleton_value(fid):
                return False
        return True

    def practice_diversity(self, coalition: FrozenSet[str]) -> int:
        """Number of distinct practices in the coalition's union."""
        union = set()
        for fid in coalition:
            union.update(self.portfolios.get(fid, set()))
        return len(union)

    def enumerate_coalitions(
        self, farmer_ids: List[str], min_size: int = 2, max_size: int = None
    ) -> List[FrozenSet[str]]:
        """Generate all coalitions of size [min_size, max_size]."""
        if max_size is None:
            max_size = len(farmer_ids)

        coalitions = []
        for k in range(min_size, max_size + 1):
            for combo in combinations(farmer_ids, k):
                coalitions.append(frozenset(combo))
        return coalitions