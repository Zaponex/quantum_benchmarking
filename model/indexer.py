import json
from typing import Dict, List, Tuple

class Indexer:
    def __init__(self):
        self._to_idx: Dict[Tuple, int] = {}
        self._from_idx: List[Tuple] = []

    def get(self, key: Tuple) -> int:
        if key in self._to_idx:
            return self._to_idx[key]
        idx = len(self._from_idx)
        self._to_idx[key] = idx
        self._from_idx.append(key)
        return idx

    def reverse(self, i: int) -> Tuple:
        return self._from_idx[i]

    def __len__(self):
        return len(self._from_idx)
    

def load_amr_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    robots = data["robots"]
    slots = data["slots"]
    tasks = data["tasks"]
    precedences = data["precedence"]

    # kleine Konsistenzprüfung
    if not (robots and slots and tasks):
        raise ValueError("robots, slots und tasks dürfen nicht leer sein.")

    return robots, slots, tasks, precedences

def assign_ent_to_indexer(indexer, amr, slots, tasks):
    x: Dict[Tuple[str, str], int] = {}
    y: Dict[Tuple[str, int], int] = {}
    w: Dict[Tuple[str, str, int], int] = {}

    # 3) Variablen registrieren
    for t in tasks:
        tname = t["name"]
        # x: Task→Robot
        for r in amr:
            x[(tname, r)] = indexer.get(("x", tname, r))
        # y: Task→Slot
        for z in slots:
            y[(tname, z)] = indexer.get(("y", tname, z))
        # w: (optional) Task→Robot→Slot
        for r in amr:
            for z in slots:
                w[(tname, r, z)] = indexer.get(("w", tname, r, z))

    return indexer, x, y, w