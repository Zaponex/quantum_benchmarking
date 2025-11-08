from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import pandas as pd

@dataclass(frozen=True)
class QuboStats:
    n_variables: int
    n_entries: int
    n_linear: int
    n_quadratic: int
    density: float

class QuboBuilder:
    def __init__(self, indexer):
        self.indexer = indexer
        self.Q: Dict[Tuple[int,int], float] = defaultdict(float)

    def add_linear(self, i: int, coeff: float) -> None:
        self.Q[(i, i)] += float(coeff)

    def add_quad(self, i: int, j: int, val: float) -> None:
        if j < i:
            i, j = j, i
        self.Q[(i, j)] += float(val)

    def prune(self, eps: float = 1e-12) -> None:
        for k in list(self.Q.keys()):
            if abs(self.Q[k]) < eps:
                del self.Q[k]

    def scale(self, factor: float) -> None:
        if factor == 1.0: 
            return
        for k in list(self.Q.keys()):
            self.Q[k] *= factor

    def as_dict(self) -> Dict[Tuple[int,int], float]:
        return dict(self.Q)

    def stats(self, size: Optional[int] = None) -> QuboStats:
        if size is None:
            size = len(self.indexer)
        n_entries = len(self.Q)
        n_linear = sum(1 for (i, j) in self.Q if i == j)
        n_quadratic = n_entries - n_linear
        max_upper = size * (size + 1) / 2 if size > 0 else 1.0
        density = n_entries / max_upper
        return QuboStats(size, n_entries, n_linear, n_quadratic, density)

    def to_dataframe(self, size: Optional[int] = None, use_labels: bool = True) -> pd.DataFrame:
        if size is None:
            size = len(self.indexer)
        mat: List[List[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
        for (i, j), v in self.Q.items():
            mat[i][j] += v
        for i in range(size):
            for j in range(i):
                mat[i][j] = mat[j][i]

        if use_labels and len(self.indexer) == size:
            idx = [self.indexer.reverse(i) for i in range(size)]
            return pd.DataFrame(mat, index=idx, columns=idx)
        return pd.DataFrame(mat)
    


    ########
    #Helper Funktion 
    ########


    def one_hot(qb, vars_idx, lam: float):
        """
        (sum v - 1)^2  ->  1*lam auf Diagonale, +2*lam auf alle Paare, und -2*lam linear insgesamt
        Expandiert: sum v_i^2  + 2*sum_{i<j} v_i v_j  - 2*sum v_i + 1
        (Konstante +lam ignorieren wir).
        """
        vars_idx = list(vars_idx)
        for i in vars_idx:
            qb.add_linear(i, lam)       
            qb.add_linear(i, -2*lam)    
        for a in range(len(vars_idx)):
            for b in range(a+1, len(vars_idx)):
                qb.add_quad(vars_idx[a], vars_idx[b], 2*lam)

    def sum_equal_sum(qb, A, B, lam: float):
        """
        (sum A - sum B)^2 = sum A^2 + sum B^2 - 2*sum_{a in A, b in B} a b + 2*sum_{i<j in A} a_i a_j + 2*sum_{i<j in B} b_i b_j - 2*sum A - 2*sum B
        (Konstante ignoriert).
        """
        A = list(A); B = list(B)
        for i in A:
            qb.add_linear(i, 1*lam)
            qb.add_linear(i, -2*lam)
        for j in B:
            qb.add_linear(j, 1*lam)
            qb.add_linear(j, -2*lam) 
        for u in range(len(A)):
            for v in range(u+1, len(A)):
                qb.add_quad(A[u], A[v], 2*lam)
        for u in range(len(B)):
            for v in range(u+1, len(B)):
                qb.add_quad(B[u], B[v], 2*lam)
        for i in A:
            for j in B:
                qb.add_quad(i, j, -2*lam)
    
    def and_link(qb, x_idx: int, y_idx: int, w_idx: int, lam: float):
        """
        W erzwingt w = x ∧ y:
        lam*(w - x)^2 + lam*(w - y)^2 + lam*(x + y - w - 1)^2   (eine übliche Form)
        Führt zu charakteristischen -2*lam Kopplungen w/x, w/y und +2*lam x/y.
        (Es gibt mehrere äquivalente Dreiterm-Varianten; diese ist stabil.)
        """
        qb.add_linear(w_idx, lam)
        qb.add_linear(x_idx, lam)
        qb.add_quad(w_idx, x_idx, -2*lam)

        qb.add_linear(w_idx, lam)
        qb.add_linear(y_idx, lam)
        qb.add_quad(w_idx, y_idx, -2*lam)

        qb.add_linear(x_idx, lam) 
        qb.add_linear(y_idx, lam)        
        qb.add_linear(w_idx, lam)      
        qb.add_quad(x_idx, y_idx, 2*lam) 
        qb.add_quad(x_idx, w_idx, -2*lam)
        qb.add_quad(y_idx, w_idx, -2*lam)
        qb.add_linear(x_idx, -2*lam)
        qb.add_linear(y_idx, -2*lam)
        qb.add_linear(w_idx,  2*lam)