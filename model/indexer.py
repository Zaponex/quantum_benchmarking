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

def count_valid_variants(robots, slots, tasks):
    """
    Berechnet nur die Anzahl der gültigen Varianten.
    Gibt eine einzige Zahl zurück.
    """
    from itertools import product
    
    # Generiere für jede Task alle (robot, start_slot) Kombinationen
    task_options = []
    for task in tasks:
        duration = task["p"]
        options = []
        for robot in robots:
            for start in slots:
                if start + duration <= max(slots) + 1:
                    options.append((robot, start, duration))
        task_options.append(options)
    
    # Zähle gültige Kombinationen
    count = 0
    for combination in product(*task_options):
        # Prüfe ob gültig (keine Überschneidungen pro Roboter)
        robot_tasks = {}
        for robot, start, duration in combination:
            if robot not in robot_tasks:
                robot_tasks[robot] = []
            robot_tasks[robot].append((start, duration))
        
        # Check overlaps
        valid = True
        for task_list in robot_tasks.values():
            for i in range(len(task_list)):
                for j in range(i + 1, len(task_list)):
                    s1, d1 = task_list[i]
                    s2, d2 = task_list[j]
                    # Overlap check
                    if not (s1 + d1 <= s2 or s2 + d2 <= s1):
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            count += 1
    
    return count