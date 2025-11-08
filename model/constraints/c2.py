from typing import List, Dict, Tuple
from model.qubo_builder import QuboBuilder

def add_assignment_exactly_one_constraints(
    qb: QuboBuilder,
    tasks: List[dict],                 # [{"name":..., "p":...}, ...] (p wird hier nicht benutzt)
    robots: List[str],                 # ["R1","R2",...]
    x: Dict[Tuple[str, str], int],     # (tname, rname) -> idx  für x_{t,r}
    lam_c2: float,                        # Strafgewicht λ
):
    """
    C2: Für jeden Task genau EIN AMR:
        H = ∑_t ( ∑_r x_{t,r} - 1 )^2

    Expansion (konstanter Term je Task ignoriert):
        (∑ v_i - 1)^2 = ∑ v_i^2 + 2∑_{i<j} v_i v_j - 2∑ v_i + 1
      -> +λ * v_i^2
         -2λ * v_i
         +2λ * v_i v_j  (für i<j)
    """
    if not lam_c2 or not tasks or not robots:
        return qb

    for t in tasks:
        tname = t["name"]
                # Explizit hier drin (reproduzierbar):
        var_indices = [x[(tname, r)] for r in robots]

        if not var_indices:
            continue

        # Diagonal/linear
        for i in var_indices:
            qb.add_linear(i, lam_c2)         # aus v_i^2 (binär)
            qb.add_linear(i, -2.0 * lam_c2)  # aus -2 * ∑ v_i

        # Paare (i<j)
        n = len(var_indices)
        for a in range(n):
            ia = var_indices[a]
            for b in range(a + 1, n):
                ib = var_indices[b]
                qb.add_quad(ia, ib, 2.0 * lam_c2)

    return qb
