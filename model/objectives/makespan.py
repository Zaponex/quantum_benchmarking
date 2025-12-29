
from typing import List, Dict
from model.qubo_builder import QuboBuilder

def add_makespan_objective(
    qb: QuboBuilder,
    tasks: List[dict],          # [{"name":..., "p":...}, ...]
    slots: List[int],           # [0,1,2,...]
    y: Dict[tuple, int],        # (tname, z) -> idx
    w_makespan: float
):
    """
    Minimiert die Summe aller Completion Times (linear):
    H₁ = Σ_t Σ_z (z + p_t) * y_tz
    
    Da QUBO quadratisch sein muss, wird dies zu:
    H₁ = Σ_t Σ_z (z + p_t) * y_tz²  (aber y_tz² = y_tz, da binär!)
    
    Args:
        qb: QUBO Builder
        tasks: Liste von Tasks mit "name" und "p" (Prozesszeit)
        slots: Verfügbare Zeitslots
        y: Mapping (task_name, slot) -> QUBO-Variable-Index
        w_makespan: Gewichtungsfaktor für dieses Objective
    
    if not w_makespan:
        return qb
    """    
    # Für jede Task: addiere (z + p_t) für jeden möglichen Slot
    for t in tasks:
        tname = t["name"]
        p = int(t["p"])
        
        for z in slots:
            completion_time = z + p
            # Linear term: (z + p) * y_tz
            # Da y² = y für binäre Variablen, ist das einfach ein linearer Term
            qb.add_linear(
                y[(tname, z)], 
                w_makespan * (completion_time **2)
            )
    
    return qb




#####OLD
"""
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
"""