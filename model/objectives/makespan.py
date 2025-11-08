from typing import List, Dict
from model.qubo_builder import QuboBuilder

def add_makespan_objective(
    qb: QuboBuilder,
    tasks: List[dict],          # [{"name":..., "p":...}, ...]
    slots: List[int],           # [0,1,2,...]
    y: Dict[tuple, int],        # (tname, z) -> idx
    w_makespan: float,
    C_ref: Dict[str,float] | None = None
):
    if not w_makespan: return
    for t in tasks:
        tname, p = t["name"], int(t["p"])
        # Diagonale
        for z in slots:
            Zi = (z + p)
            qb.add_linear(y[(tname,z)], w_makespan * (Zi**2))
        # Off-Diagonalen y-y
        for i, z1 in enumerate(slots):
            Zi = (z1 + p)
            for z2 in slots[i+1:]:
                Zj = (z2 + p)
                qb.add_quad(y[(tname,z1)], y[(tname,z2)], 2.0*w_makespan*Zi*Zj)
        # Referenz
        if C_ref and tname in C_ref:
            cref = float(C_ref[tname])
            for z in slots:
                Zi = (z + p)
                qb.add_linear(y[(tname,z)], -2.0*w_makespan*cref*Zi)


    return qb
