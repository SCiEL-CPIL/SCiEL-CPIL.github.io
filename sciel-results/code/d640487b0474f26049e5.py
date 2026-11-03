"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _compute_score(U):
    """Compute the score for a set U."""
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    # Compute U+U and U-U
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    import math
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def _greedy_construct(deadline):
    """Greedy construction of set U."""
    U = [0]
    # Try adding elements in increasing order, choosing the one that maximizes score
    best_score = _compute_score(U)
    max_val = 50  # Limit search space for greedy
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    
    while time.time() < deadline and len(U) < 30:
        best_candidate = None
        best_new_score = best_score
        for c in candidates:
            if c in U:
                continue
            new_U = U + [c]
            s = _compute_score(new_U)
            if s > best_new_score:
                best_new_score = s
                best_candidate = c
        if best_candidate is None:
            break
        U.append(best_candidate)
        best_score = best_new_score
        candidates = [c for c in candidates if c not in U]
        if not candidates:
            break
    return U, best_score

def _local_search(U, deadline):
    """Local search to improve the set."""
    U = list(U)
    best_score = _compute_score(U)
    max_val = 60
    all_candidates = list(range(0, max_val + 1))
    
    while time.time() < deadline:
        improved = False
        # Try adding an element
        for c in all_candidates:
            if c in U:
                continue
            new_U = U + [c]
            s = _compute_score(new_U)
            if s > best_score:
                U = new_U
                best_score = s
                improved = True
                break
        if improved:
            continue
        # Try removing an element (except 0)
        for i, u in enumerate(U):
            if u == 0:
                continue
            new_U = U[:i] + U[i+1:]
            if not new_U or 0 not in new_U:
                continue
            s = _compute_score(new_U)
            if s > best_score:
                U = new_U
                best_score = s
                improved = True
                break
        if improved:
            continue
        # Try swapping an element
        for i, u in enumerate(U):
            if u == 0:
                continue
            for c in all_candidates:
                if c in U:
                    continue
                new_U = U[:i] + [c] + U[i+1:]
                s = _compute_score(new_U)
                if s > best_score:
                    U = new_U
                    best_score = s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # Initial greedy construction
    random.seed(42)
    U, score = _greedy_construct(deadline)
    
    # Local search refinement
    if time.time() < deadline:
        U, score = _local_search(U, deadline)
    
    return U
# EVOLVE-BLOCK-END

