import numpy as np
import time
import random
from itertools import combinations

def _squared_norm(v):
    return int(np.dot(v, v))

def _squared_dist(a, b):
    d = a - b
    return int(np.dot(d, d))

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 585.0  # safety margin before 630s kill

    best_points = None
    best_k = 0

    # Start with the known 220-point solution (weight-2 vectors)
    base_points = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    base_points.append(p)
    base_points = np.array(base_points, dtype=np.int64)
    best_points = base_points.copy()
    best_k = len(base_points)  # 220

    def generate_candidates(weights):
        cands = []
        for w in weights:
            for combo in combinations(range(d), w):
                for mask in range(2 ** w):
                    v = np.zeros(d, dtype=np.int64)
                    for idx, pos in enumerate(combo):
                        if mask & (1 << idx):
                            v[pos] = 1
                        else:
                            v[pos] = -1
                    cands.append(v)
        return np.array(cands, dtype=np.int64)

    # Try different weight combinations
    weight_sets = [
        [2],           # 220 candidates - the base
        [1, 2],        # 242 candidates
        [2, 3],        # 1540 candidates
        [1, 2, 3],     # 1562 candidates
        [2, 3, 4],     # 6820 candidates
        [1, 2, 3, 4],  # 6842 candidates
    ]

    rng = random.Random(42)

    for ws in weight_sets:
        if time.time() > deadline - 15:
            break

        cands = generate_candidates(ws)
        n = len(cands)

        # Precompute norms
        norms = np.array([_squared_norm(c) for c in cands], dtype=np.int64)

        # Greedy selection with multiple strategies
        max_restarts = 30

        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            # Different ordering strategies based on restart number
            if restart % 3 == 0:
                # Random shuffle
                order = list(range(n))
                rng.shuffle(order)
            elif restart % 3 == 1:
                # Sort by norm (ascending)
                order = sorted(range(n), key=lambda i: (norms[i], i))
            else:
                # Sort by norm (descending)
                order = sorted(range(n), key=lambda i: (-norms[i], i))

            selected = []
            selected_norms = []
            min_dist_sq = float('inf')

            for idx in order:
                if time.time() > deadline - 10:
                    break

                p = cands[idx]
                pn = norms[idx]

                # Check against all selected points
                compatible = True
                current_max_norm = max(selected_norms) if selected else 0
                required_min_dist = max(min_dist_sq if min_dist_sq != float('inf') else 0, current_max_norm, pn)

                for sp in selected:
                    d_sq = _squared_dist(p, sp)
                    if d_sq < required_min_dist:
                        compatible = False
                        break
                    if d_sq < min_dist_sq:
                        min_dist_sq = d_sq

                if compatible:
                    selected.append(p)
                    selected_norms.append(pn)

            if len(selected) > best_k:
                best_k = len(selected)
                best_points = np.array(selected, dtype=np.int64)

    # Also try the pure weight-2 solution as a fallback
    if best_points is None:
        best_points = base_points
        best_k = 220

    return np.array(best_points, dtype=np.int64)

# EVOLVE-BLOCK-END
