from typing import List, Dict, Tuple
from model.qubo_builder import QuboBuilder

def add_c3_capacity_no_overlap_inline(
    qb: QuboBuilder,
    tasks: List[dict],                     # [{"name":..., "p":...}, ...]
    robots: List[str],                     # ["R1","R2",...]
    slots: List[int],                      # [0,1,2,...]
    x: Dict[Tuple[str, str], int],         # (tname, rname) -> idx  für x_{t,r}
    y: Dict[Tuple[str, int], int],         # (tname, z)     -> idx  für y_{t,z}
    w: Dict[Tuple[str, str, int], int],    # (tname, rname, z) -> idx für w_{t,r,z}
    lam3_and: float,                        # λ für AND-Verknüpfung
    lam3_cap: float,                        # λ für Kapazität (Exactly-One über w pro (r,z))
):
    """
    c3 = H5 = ∑_{z∈Z} ∑_{r∈R} ( ∑_{t∈T} w_{t,r,z} - 1 )²
    mit w_{t,r,z} = x_{t,r} ∧ y_{t,z}

    Modellbeschreibung:
    -------------------
    Dieser Term stellt die **Kapazitäts- bzw. No-Overlap-Bedingung** sicher.
    Er sorgt dafür, dass zu jedem Zeitpunkt z auf einem Roboter r **höchstens eine**
    Aufgabe aktiv ist. Dazu wird für jedes (r,z)-Paar eine Exactly-One-Bedingung über
    die Hilfsvariablen w_{t,r,z} eingeführt.

    Die w-Variablen werden über AND-Verknüpfungen zwischen den ursprünglichen
    Entscheidungsvariablen x_{t,r} (Task–Roboter-Zuordnung) und y_{t,z}
    (Task–Startslot) eingeführt:

        w_{t,r,z} = x_{t,r} ∧ y_{t,z}

    Damit wird ausgedrückt: "Task t wird genau dann zur Zeit z von Roboter r ausgeführt,
    wenn t Roboter r zugewiesen und in Zeitslot z gestartet ist."

    Die AND-Verknüpfung wird über quadratische Penalty-Terme abgebildet, welche die
    Beziehung zwischen den Variablen energetisch erzwingen:

        (w - x)² + (w - y)² + (x + y - w - 1)²

    Diese Kombination gewährleistet, dass w=1 nur gilt, wenn sowohl x=1 als auch y=1
    sind. Abweichungen davon werden durch das Strafgewicht λ_and bestraft.

    Zusätzlich erzwingt der Kapazitätsterm (Exactly-One über w pro (r,z)):

        (∑_{t∈T} w_{t,r,z} - 1)²

    dass zu jedem Zeitpunkt z auf jedem Roboter r nur eine Aufgabe aktiv ist.
    Dadurch wird ein **zeitlicher Überlapp von Aufgaben auf demselben Roboter**
    verhindert.

    Parameter:
    ----------
    lam_and : float
        Lagrange-Faktor für die Konsistenzbedingung (AND-Verknüpfung).
        Höherer Wert → stärkeres Erzwingen der logischen Kopplung.
    lam_cap : float
        Lagrange-Faktor für die Kapazitätsbedingung (Exactly-One).
        Höherer Wert → stärkeres Erzwingen der Nicht-Überlappung.
    """    
    # --- eingebettete Hilfsfunktionen -------------------------------
    def _and_link(xi: int, yi: int, wi: int, lam: float):
        # (w - x)^2 = w + x - 2 w x
        qb.add_linear(wi, lam)
        qb.add_linear(xi, lam)
        qb.add_quad(wi, xi, -2 * lam)
        # (w - y)^2 = w + y - 2 w y
        qb.add_linear(wi, lam)
        qb.add_linear(yi, lam)
        qb.add_quad(wi, yi, -2 * lam)
        # (x + y - w - 1)^2 = x + y + w + 2xy - 2xw - 2yw - 2x - 2y + 2w + 1
        qb.add_linear(xi, lam)
        qb.add_linear(yi, lam)
        qb.add_linear(wi, lam)
        qb.add_quad(xi, yi, 2 * lam)
        qb.add_quad(xi, wi, -2 * lam)
        qb.add_quad(yi, wi, -2 * lam)
        qb.add_linear(xi, -2 * lam)
        qb.add_linear(yi, -2 * lam)
        qb.add_linear(wi,  2 * lam)

    def _one_hot(vars_: List[int], lam: float):
        # (Σ v - 1)^2 = Σ v_i^2 + 2 Σ_{i<j} v_i v_j - 2 Σ v_i + 1
        if not vars_ or not lam:
            return
        for v_i in vars_:
            qb.add_linear(v_i, lam)         # Σ v_i^2  (binär)
        for a in range(len(vars_)):
            for b in range(a + 1, len(vars_)):
                qb.add_quad(vars_[a], vars_[b], 2 * lam)  # 2 Σ_{i<j} v_i v_j
        for v_i in vars_:
            qb.add_linear(v_i, -2 * lam)    # -2 Σ v_i
    # ----------------------------------------------------------------

    if not tasks or not robots or not slots:
        return qb

    # AND-Verknüpfungen: w_{t,r,z} = x_{t,r} ∧ y_{t,z}
    if lam3_and:
        for t in tasks:
            tname = t["name"]
            for r in robots:
                for z in slots:
                    _and_link(
                        x[(tname, r)],
                        y[(tname, z)],
                        w[(tname, r, z)],
                        lam3_and
                    )

    # Kapazität pro (r,z): Exactly-One über w_{t,r,z}
    if lam3_cap:
        for r in robots:
            for z in slots:
                _one_hot([w[(t["name"], r, z)] for t in tasks], lam3_cap)

    return qb
