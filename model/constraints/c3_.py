from typing import List, Dict, Tuple
from model.qubo_builder import QuboBuilder

def add_c3_simplified(
    qb: QuboBuilder,
    tasks: List[dict],
    robots: List[str],
    slots: List[int],
    x: Dict[Tuple[str, str], int],
    y: Dict[Tuple[str, int], int],
    w: Dict[Tuple[str, str, int], int],
    lam_c3_and: float,
    lam_c3_cap: float,
):
    """
    C3 Simplified - Capacity & No-Overlap OHNE Duration-Constraint
    
    Erzwingt:
      1) w_{t,r,z} ≤ x_{t,r}  (w kann nur an, wenn Task t auf Robot r)
      2) w_{t,r,z} ≤ Σ_{s: s≤z<s+p} y_{t,s}  (w nur im Zeitfenster aktiv)
      3) ❌ ENTFERNT: Σ_z w_{t,r,z} = p_t · x_{t,r}  (Duration-Constraint)
      4) At-most-one pro (r,z): Σ_t w_{t,r,z} ≤ 1  (keine Überlappung)
    
    UNTERSCHIED zur Vollversion:
    - Keine w-w Kopplungen durch Duration! (420 Terme weniger)
    - Duration wird nur "weich" durch Fenster-Links erzwungen
    - Task KANN mehr/weniger Slots belegen (aber Objective + Fenster machen es unwahrscheinlich)
    
    Args:
        qb: QUBO Builder
        tasks: Liste von Tasks mit "name" und "p" (Prozesszeit)
        robots: Liste der verfügbaren Roboter
        slots: Verfügbare Zeitslots
        x: Mapping (task, robot) -> QUBO-Variable-Index
        y: Mapping (task, slot) -> QUBO-Variable-Index
        w: Mapping (task, robot, slot) -> QUBO-Variable-Index
        lam_c3_and: Gewicht für Linking-Constraints (1-2)
        lam_c3_cap: Gewicht für Capacity-Constraint (4)
    """

    if not tasks or not robots or not slots:
        return qb

    dur = {t["name"]: t["p"] for t in tasks}

    # ═══════════════════════════════════════════════════════════
    #  HELPER FUNCTIONS
    # ═══════════════════════════════════════════════════════════
    
    def _at_most_one(vars_, lam):
        """At-most-one constraint: Σ_i v_i ≤ 1 via pairwise penalties."""
        if not vars_ or not lam:
            return
        for a in range(len(vars_)):
            for b in range(a + 1, len(vars_)):
                qb.add_quad(vars_[a], vars_[b], 2 * lam)

    def _link_w_le_x(wi, xi, lam):
        """Upper bound: w ≤ x via penalty w(1-x)"""
        qb.add_linear(wi, lam)
        qb.add_quad(wi, xi, -lam)

    def _link_w_le_window(wi, y_idxs, lam):
        """Upper bound: w ≤ Σy via penalty w(1 - Σy)"""
        if not y_idxs:
            # Kein gültiges Fenster -> w muss 0 sein
            qb.add_linear(wi, lam)
            return
        
        # Standard: w(1 - Σy)
        qb.add_linear(wi, lam)
        for yi in y_idxs:
            qb.add_quad(wi, yi, -lam)

    # ═══════════════════════════════════════════════════════════
    #  LINKING (lam_c3_and) - OHNE Duration!
    # ═══════════════════════════════════════════════════════════
    
    if lam_c3_and:
        for t in tasks:
            tname = t["name"]
            p = dur[tname]
            
            for r in robots:
                xi = x[(tname, r)]
                
                for z in slots:
                    wi = w[(tname, r, z)]

                    # 1) Upper bound: w_{t,r,z} ≤ x_{t,r}
                    _link_w_le_x(wi, xi, lam_c3_and)

                    # 2) Upper bound: w_{t,r,z} ≤ Σ_{s: s≤z<s+p} y_{t,s}
                    #    (z ist aktiv, wenn Task in [z-p+1, ..., z] startet)
                    window_s = [s for s in slots if z - p + 1 <= s <= z]
                    y_idxs = [y[(tname, s)] for s in window_s]
                    _link_w_le_window(wi, y_idxs, lam_c3_and)

                # ❌ ENTFERNT: Duration-Constraint
                # _eq_sum_w_equals_p_times_x(w_all_z, xi, p, lam_c3_and)

    # ═══════════════════════════════════════════════════════════
    #  CAPACITY (lam_c3_cap)
    # ═══════════════════════════════════════════════════════════
    
    # 4) Capacity: At-most-one Task pro (Robot, Slot)
    if lam_c3_cap:
        for r in robots:
            for z in slots:
                conflict_vars = [w[(t["name"], r, z)] for t in tasks]
                _at_most_one(conflict_vars, lam_c3_cap)

    return qb


# ═══════════════════════════════════════════════════════════
#  ORIGINAL (FULL VERSION) - Zum Vergleich behalten
# ═══════════════════════════════════════════════════════════

def add_c3_capacity_no_overlap(
    qb: QuboBuilder,
    tasks: List[dict],
    robots: List[str],
    slots: List[int],
    x: Dict[Tuple[str, str], int],
    y: Dict[Tuple[str, int], int],
    w: Dict[Tuple[str, str, int], int],
    lam_c3_and: float,
    lam_c3_cap: float,
):
    """
    VOLLVERSION mit Duration-Constraint (erzeugt 420 w-w Kopplungen!)
    
    Erzwingt zusätzlich:
      3) Σ_z w_{t,r,z} = p_t · x_{t,r}  (Task belegt GENAU p Slots)
    """
    
    if not tasks or not robots or not slots:
        return qb

    dur = {t["name"]: t["p"] for t in tasks}

    def _at_most_one(vars_, lam):
        if not vars_ or not lam:
            return
        for a in range(len(vars_)):
            for b in range(a + 1, len(vars_)):
                qb.add_quad(vars_[a], vars_[b], 2 * lam)

    def _link_w_le_x(wi, xi, lam):
        qb.add_linear(wi, lam)
        qb.add_quad(wi, xi, -lam)

    def _link_w_le_window(wi, y_idxs, lam):
        if not y_idxs:
            qb.add_linear(wi, lam)
            return
        qb.add_linear(wi, lam)
        for yi in y_idxs:
            qb.add_quad(wi, yi, -lam)

    def _eq_sum_w_equals_p_times_x(w_idxs, xi, p, lam):
        """
        ⚠️ ERZEUGT w-w KOPPLUNGEN!
        (Σ_z w - p·x)² = (Σw)² - 2p·x·Σw + p²·x
        """
        if not w_idxs or not lam:
            return
        
        # (Σw)²: erzeugt w-w Kopplungen!
        for i in range(len(w_idxs)):
            qb.add_linear(w_idxs[i], lam)
            for j in range(i + 1, len(w_idxs)):
                qb.add_quad(w_idxs[i], w_idxs[j], 2 * lam)  # ← 420 w-w Terme!
        
        # -2p·x·Σw
        for wi in w_idxs:
            qb.add_quad(wi, xi, -2 * p * lam)
        
        # +p²·x
        qb.add_linear(xi, (p * p) * lam)

    # Linking + Duration
    if lam_c3_and:
        for t in tasks:
            tname = t["name"]
            p = dur[tname]
            
            for r in robots:
                xi = x[(tname, r)]
                w_all_z = []
                
                for z in slots:
                    wi = w[(tname, r, z)]
                    w_all_z.append(wi)

                    _link_w_le_x(wi, xi, lam_c3_and)

                    window_s = [s for s in slots if z - p + 1 <= s <= z]
                    y_idxs = [y[(tname, s)] for s in window_s]
                    _link_w_le_window(wi, y_idxs, lam_c3_and)

                # ⚠️ Duration-Constraint (erzeugt w-w!)
                _eq_sum_w_equals_p_times_x(w_all_z, xi, p, lam_c3_and)

    # Capacity
    if lam_c3_cap:
        for r in robots:
            for z in slots:
                conflict_vars = [w[(t["name"], r, z)] for t in tasks]
                _at_most_one(conflict_vars, lam_c3_cap)

    return qb