from typing import List, Dict, Tuple
from model.qubo_builder import QuboBuilder

def add_c5_precedence_inline(
    qb: QuboBuilder,
    tasks: List[dict],                      # [{"name":..., "p":...}, ...]
    slots: List[int],                       # [0,1,2,...]
    y: Dict[Tuple[str, int], int],          # (tname, z) -> idx  für y_{t,z}
    precedence: List[Tuple[str, str]],      # Paare (a,b) mit a ≺ b
    lam_c5: float,                          # λ für c5 (Präzedenz)
):
    """
    c5 = H7 = ∑_{(a,b)∈P} ∑_{z_a∈Z} ∑_{z_b < z_a + p_a} y_{a,z_a} * y_{b,z_b}

    Bestraft alle Startzeit-Kombinationen, bei denen b vor dem Ende von a startet.
    (Nur Quadratterme; keine linearen Beiträge.)
    """
    if not lam_c5 or not tasks or not slots or not precedence:
        return qb

    # Dauer p_t pro Task
    p_by_t = {t["name"]: int(t["p"]) for t in tasks}

    for (a, b) in precedence:
        p_a = p_by_t[a]
        for z_a in slots:
            for z_b in slots:
                if z_b < z_a + p_a:  # Verletzung: b startet zu früh
                    qb.add_quad(y[(a, z_a)], y[(b, z_b)], lam_c5)

    return qb
