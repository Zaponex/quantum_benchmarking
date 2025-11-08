from typing import List, Dict
from model.qubo_builder import QuboBuilder

def add_workload_balance_objective(
    qb: QuboBuilder,
    tasks: List[dict],            # [{"name":..., "p":...}, ...]
    robots: List[str],            # ["R1","R2",...]
    x: Dict[tuple, int],          # (tname, rname) -> idx  für x_{t,r}
    w_balance: float,
):
    """
    H2 = sum_r S_r^2 - (1/R) * S_tot^2
    mit S_r = sum_t p_t * x_{t,r}

    Entstehende Terme (auf x):
      Linear:                (1 - 1/R) * p_t^2
      Quad (gleiches r):     2 * (1 - 1/R) * p_t * p_u
      Quad (versch. r):     -2 * (1/R)     * p_t * p_u
    Alle Terme werden mit w_balance gewichtet.
    """
    if not w_balance:
        return qb
    if not tasks or not robots:
        return qb

    R = float(len(robots))
    invR = 1.0 / R
    one_minus_invR = 1.0 - invR


    p_by_t = {t["name"]: float(t["p"]) for t in tasks}
    task_names = list(p_by_t.keys())

    for r in robots:
        for tname in task_names:
            i = x[(tname, r)]
            qb.add_linear(i, w_balance * one_minus_invR * (p_by_t[tname] ** 2))

    for r in robots:
        for idx1 in range(len(task_names)):
            t1 = task_names[idx1]
            p1 = p_by_t[t1]
            for idx2 in range(idx1 + 1, len(task_names)):
                t2 = task_names[idx2]
                p2 = p_by_t[t2]
                i = x[(t1, r)]
                j = x[(t2, r)]
                qb.add_quad(i, j, 2.0 * w_balance * one_minus_invR * (p1 * p2))

    for r_idx1 in range(len(robots)):
        r1 = robots[r_idx1]
        for r_idx2 in range(r_idx1 + 1, len(robots)):
            r2 = robots[r_idx2]
            for t1 in task_names:
                p1 = p_by_t[t1]
                for t2 in task_names:
                    p2 = p_by_t[t2]
                    i = x[(t1, r1)]
                    j = x[(t2, r2)]
                    qb.add_quad(i, j, -2.0 * w_balance * invR * (p1 * p2))

    return qb