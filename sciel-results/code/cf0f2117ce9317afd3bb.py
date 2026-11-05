"""Direct approach for minimum maximal no-isosceles set.
Combines bitset-based constraint propagation (from current) with
deletion-based minimization (from inspiration 1) and a
hypergraph-style branch-and-bound flavor for local refinement.
"""
N = 16
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
TOTAL = N * N

def _point_idx(p):
    return p[0] * N + p[1]

def _idx_point(i):
    return (i // N, i % N)

# Precompute iso_third[i][j] = bool array length TOTAL: c such that {i,j,c} isosceles
def _precompute():
    pts = [_idx_point(i) for i in range(TOTAL)]
    iso = np.zeros((TOTAL, TOTAL, TOTAL), dtype=bool)
    for i in range(TOTAL):
        for j in range(TOTAL):
            if i == j:
                continue
            a = pts[i]; b = pts[j]
            da = (a[0]-b[0])**2 + (a[1]-b[1])**2
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                c = pts[k]
                db = (a[0]-c[0])**2 + (a[1]-c[1])**2
                dc = (b[0]-c[0])**2 + (b[1]-c[1])**2
                if da == db or da == dc or db == dc:
                    iso[i, j, k] = True
    return iso

_iso_third = _precompute()

# Precompute for each point p, the set of points q such that {p, q, c} isosceles for some c.
# Actually, for greedy: when we add p, we need to block all c such that exists q in chosen with iso[p][q][c].
# Let's precompute for each p, a list of (q, c) pairs where iso[p][q][c] is True.
# Or better: for each p, precompute a 2D boolean array iso[p][q][c].

def _greedy(order):
    """order: list of point indices. Returns (chosen_list, blocked_mask)."""
    chosen = []
    blocked = np.zeros(TOTAL, dtype=bool)
    chosen_set = set()
    for p in order:
        if blocked[p]:
            continue
        # Add p: block all c such that exists q in chosen with {p,q,c} iso
        # For each q in chosen, find all c where iso_third[p][q][c] is True
        # We can do this efficiently by precomputing for each p, the union over q in chosen of iso_third[p][q]
        # But that's expensive. Instead, for each q in chosen, we look at iso_third[p][q] which is a bool array of length TOTAL
        # We only need to block c where iso_third[p][q][c] is True AND c is not p, q
        # Let's just iterate
        new_blocked = np.zeros(TOTAL, dtype=bool)
        for q in chosen:
            # iso_third[p][q] is a bool array of length TOTAL
            # Block all c where this is True
            new_blocked |= _iso_third[p][q]
        # Don't block p itself or any chosen point (they are already in the set)
        new_blocked[p] = False
        for q in chosen:
            new_blocked[q] = False
        blocked |= new_blocked
        blocked[p] = True
        chosen.append(p)
    return chosen, blocked

def _is_maximal(chosen, blocked):
    cmask = np.zeros(TOTAL, dtype=bool)
    cmask[chosen] = True
    unchosen = ~cmask
    # maximal iff no unchosen point is unblocked
    return not (blocked[unchosen] == False).any()

def _minimize_by_deletion(chosen, all_points):
    """Repeatedly remove points if the set remains maximal after removal."""
    chosen = list(chosen)
    changed = True
    while changed:
        changed = False
        for i in range(len(chosen)):
            # Try removing chosen[i]
            reduced = chosen[:i] + chosen[i+1:]
            if _is_maximal_set(reduced, all_points):
                chosen = reduced
                changed = True
                break  # restart loop since list changed
    return chosen

def _is_maximal_set(chosen, all_points):
    """Check if the set is maximal: no point outside can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True

def _can_add(p, chosen):
    """Check if p can be added to chosen without forming isosceles with any pair."""
    for i in range(len(chosen)):
        for j in range(i+1, len(chosen)):
            if _isosceles(p, chosen[i], chosen[j]):
                return False
    return True

def _isosceles(a, b, c):
    d1 = _sq_dist(a, b)
    d2 = _sq_dist(a, c)
    d3 = _sq_dist(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _sq_dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def construct():
    all_idx = list(range(TOTAL))
    all_points = [_idx_point(i) for i in all_idx]
    best = None
    best_size = TOTAL + 1
    start = time.time()
    deadline = start + 320  # safety margin before 330s kill

    random.seed(42)
    # First try a deterministic ordering (row-major) as baseline
    order = all_idx[:]
    chosen, blocked = _greedy(order)
    if _is_maximal(chosen, blocked):
        # Now try to minimize by deletion
        minimized = _minimize_by_deletion(chosen, all_points)
        if len(minimized) < best_size:
            best_size = len(minimized)
            best = minimized

    for attempt in range(50000):
        if time.time() > deadline:
            break
        order = all_idx[:]
        random.shuffle(order)
        chosen, blocked = _greedy(order)
        if _is_maximal(chosen, blocked):
            # Now try to minimize by deletion
            minimized = _minimize_by_deletion(chosen, all_points)
            if len(minimized) < best_size:
                best_size = len(minimized)
                best = minimized
                if best_size <= 10:
                    break

    if best is None:
        best = all_idx[:]
    return [_idx_point(i) for i in best]

# EVOLVE-BLOCK-END
