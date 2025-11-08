from typing import List, Dict, Tuple
from model.qubo_builder import QuboBuilder

def add_c4_consistency_inline(
    qb: QuboBuilder,
    tasks: List[dict],                  # [{"name":..., "p":...}, ...]
    robots: List[str],                  # ["R1","R2",...]
    slots: List[int],                   # [0,1,2,...]
    x: Dict[Tuple[str, str], int],      # (tname, rname) -> idx  für x_{t,r}
    y: Dict[Tuple[str, int], int],      # (tname, z)     -> idx  für y_{t,z}
    lam_c4: float,                      # Strafgewicht λ
):
    """
    c4 = H6 = ∑_{t∈T} ( ∑_{r∈R} x_{t,r}  -  ∑_{z∈Z} y_{t,z} )²

    Jede Aufgabe wird genau einem Zeitslot zugewiesen.

    Erweiterte Form:
        (Σx - Σy)² = Σx² + Σy² + 2Σ_{r<r'} x_r x_r' + 2Σ_{z<z'} y_z y_z' - 2Σ_{r,z} x_r y_z
    """

    if not lam_c4 or not tasks or not robots or not slots:
        return qb

    for t in tasks:
        tname = t["name"]

        # Indizes für diesen Task (explizit im Funktionskörper)
        x_idx = [x[(tname, r)] for r in robots]
        y_idx = [y[(tname, z)] for z in slots]

        # --- (Σ x)²: lineare und Paar-Terme
        for xi in x_idx:
            qb.add_linear(xi, lam_c4)          # x_i² → linear bei binär
        for a in range(len(x_idx)):
            for b in range(a + 1, len(x_idx)):
                qb.add_quad(x_idx[a], x_idx[b], 2 * lam_c4)

        # --- (Σ y)²: lineare und Paar-Terme
        for yi in y_idx:
            qb.add_linear(yi, lam_c4)
        for a in range(len(y_idx)):
            for b in range(a + 1, len(y_idx)):
                qb.add_quad(y_idx[a], y_idx[b], 2 * lam_c4)

        # --- Kreuzterm -2 Σ_{r,z} x_r y_z
        for xi in x_idx:
            for yi in y_idx:
                qb.add_quad(xi, yi, -2 * lam_c4)

    return qb
