"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
N2 = N * N

# EVOLVE-BLOCK-START
import itertools
import random
import time

# Precompute point coordinates and index mapping
points = list(itertools.product(range(N), repeat=2))
point_to_idx = {p: i for i, p in enumerate(points)}

# Precompute squared distances
dist2 = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    xi, yi = points[i]
    for j in range(i + 1, N2):
        xj, yj = points[j]
        dx = xi - xj
        dy = yi - yj
        d = dx * dx + dy * dy
        dist2[i][j] = d
        dist2[j][i] = d

def _is_iso_idx(i, j, k):
    d1 = dist2[i][j]
    d2 = dist2[i][k]
    d3 = dist2[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3

def is_maximal(S):
    """
    S: list of indices
    """
    S_set = set(S)
    # Check each point not in S
    for q in range(N2):
        if q in S_set:
            continue
        blocked = False
        # Check all pairs in S
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                if _is_iso_idx(q, a, b):
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_size = float('inf')
    best_set = []
    
    def greedy_construct(order):
        chosen = []
        for p in order:
            valid = True
            for i in range(len(chosen)):
                a = chosen[i]
                for j in range(i + 1, len(chosen)):
                    b = chosen[j]
                    if _is_iso_idx(p, a, b):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                chosen.append(p)
        return chosen

    def shrink(S):
        S = list(S)
        changed = True
        while changed:
            changed = False
            indices = list(range(len(S)))
            random.shuffle(indices)
            for idx in indices:
                p = S[idx]
                candidate = S[:idx] + S[idx + 1:]
                if is_maximal(candidate):
                    S = candidate
                    changed = True
                    break
        return S

    # Run restarts
    while time.time() < deadline:
        order = list(range(N2))
        random.shuffle(order)
        
        S = greedy_construct(order)
        S = shrink(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
    # Convert indices back to points
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
