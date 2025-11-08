from typing import List, Dict, Tuple
from model.qubo_builder import QuboBuilder

def add_c3_capacity_no_overlap_inline(
    qb: QuboBuilder,
    tasks: List[dict],                     # [{"name":..., "p":...}]
    robots: List[str],                     # ["R1","R2",...]
    slots: List[int],                      # [0,1,2,...]
    x: Dict[Tuple[str, str], int],         # (tname, rname) -> idx  für x_{t,r}
    y: Dict[Tuple[str, int], int],         # (tname, z)     -> idx  für y_{t,s}
    w: Dict[Tuple[str, str, int], int],    # (tname, rname, z) -> idx für w_{t,r,z}
    lam3_and: float,                       # Link-/Fenster-/Zähl-Gewicht
    lam3_cap: float,                       # Kapazität (at-most-one) pro (r,z)
):
    """
    Erzwingt korrekt:
      1) w_{t,r,z} darf nur an, wenn x_{t,r}=1 und z im Fenster eines Startslots liegt
         (ohne zusätzliche Hilfsvariablen, "Upper-Bound"-Links).
      2) Sum_z w_{t,r,z} = p_t * x_{t,r}  (richtige Dauer pro Task/Roboter)
      3) Kapazität: at-most-one pro (r,z) über alle Tasks.
    """

    if not tasks or not robots or not slots:
        return qb

    dur = {t["name"]: t["p"] for t in tasks}

    # ---- kleine Helfer ----
    def _at_most_one(vars_, lam):
        # Bestraft nur Kollisionen; Leerlauf ist erlaubt
        if not vars_ or not lam:
            return
        for a in range(len(vars_)):
            for b in range(a + 1, len(vars_)):
                qb.add_quad(vars_[a], vars_[b], 2 * lam)

    def _link_w_le_x(wi, xi, lam):
        # w <= x  -> Strafterm: w*(1 - x)
        qb.add_linear(wi, lam)
        qb.add_quad(wi, xi, -lam)

    def _link_w_le_window(wi, y_idxs, lam):
        # w <= sum_s y_s  -> Strafterm: w*(1 - sum_s y_s)
        qb.add_linear(wi, lam)
        for yi in y_idxs:
            qb.add_quad(wi, yi, -lam)

    def _eq_sum_w_equals_p_times_x(w_idxs, xi, p, lam):
        # (sum_z w - p*x)^2
        # = (sum w)^2 - 2p x sum w + p^2 x
        if not w_idxs or not lam:
            return
        # (sum w)^2
        for i in range(len(w_idxs)):
            qb.add_linear(w_idxs[i], lam)  # w_i^2
            for j in range(i + 1, len(w_idxs)):
                qb.add_quad(w_idxs[i], w_idxs[j], 2 * lam)
        # - 2 p x sum w
        for wi in w_idxs:
            qb.add_quad(wi, xi, -2 * p * lam)
        # + p^2 x
        qb.add_linear(xi, (p * p) * lam)
    # ------------------------

    # Fenster-Links + Zählgleichheit (nutzt lam3_and)
    if lam3_and:
        for t in tasks:
            tname = t["name"]
            p = dur[tname]
            for r in robots:
                xi = x[(tname, r)]
                w_all_z = []
                for z in slots:
                    wi = w[(tname, r, z)]
                    w_all_z.append(wi)

                    # w <= x
                    _link_w_le_x(wi, xi, lam3_and)

                    # w <= sum_{s: s <= z < s+p} y_{t,s}
                    window_s = [s for s in slots if s <= z < s + p]
                    y_idxs = [y[(tname, s)] for s in window_s]
                    _link_w_le_window(wi, y_idxs, lam3_and)

                # Sum_z w = p * x  (erzwingt genau p belegte Slots, wenn Task aktiv)
                _eq_sum_w_equals_p_times_x(w_all_z, xi, p, lam3_and)

    # Kapazität pro (r,z): at-most-one über w_{t,r,z}
    if lam3_cap:
        for r in robots:
            for z in slots:
                _at_most_one([w[(t["name"], r, z)] for t in tasks], lam3_cap)

    return qb
