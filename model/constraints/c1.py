from typing import List, Dict, Tuple
from model.qubo_builder import QuboBuilder

def add_startslot_exactly_one_constraints(
    qb: QuboBuilder,
    tasks: List[dict],                 # [{"name":..., "p":...}, ...]  (p wird hier nicht benutzt)
    slots: List[int],                  # [0,1,2,...]
    y: Dict[Tuple[str, int], int],     # (tname, z) -> idx  für y_{t,z}
    lam_c1: float,                         # Strafgewicht λ
):
    """
    C1: Für jeden Task genau EIN Startslot:
        H_3 = ∑_t ( ∑_z y_{t,z} - 1 )^2
    Expansion (konstante +λ je Task ignoriert):
        +λ * v_i^2  - 2λ * v_i  + 2λ * v_i v_j  (für i<j)
    """
    if not lam_c1 or not tasks or not slots:
        return qb

    for t in tasks:
        tname = t["name"]
        
        # explizit hier drin (reproduzierbar):
        var_indices = [y[(tname, z)] for z in slots]

        if not var_indices:
            continue

        # Diagonal/linear
        for i in var_indices:
            qb.add_linear(i, lam_c1)         # aus v_i^2 (binär)
            qb.add_linear(i, -2.0 * lam_c1)  # aus -2 * sum v_i

        # Paare
        n = len(var_indices)
        for a in range(n):
            
            ia = var_indices[a]
            for b in range(a + 1, n):
                ib = var_indices[b]
                qb.add_quad(ia, ib, 2.0 * lam_c1)

    return qb