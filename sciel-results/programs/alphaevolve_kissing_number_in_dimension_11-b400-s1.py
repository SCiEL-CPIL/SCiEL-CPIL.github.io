"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s limit
    
    # Generate all weight-4 vectors with ±1 entries
    # C(11,4) = 330 positions, 2^4 = 16 sign combos -> 5280 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Norm squared of each candidate = 4
    # Constraint: |u-v|^2 >= 4 for all pairs, i.e., u·v <= 2
    
    def greedy_select(order, time_limit):
        """Greedy selection following the given order."""
        selected = []
        for idx in order:
            if time.time() > time_limit:
                break
            v = candidates[idx]
            if not selected:
                selected.append(v)
                continue
            # Constraint: u·v <= 2 for all selected u
            dots = np.dot(selected, v)
            if np.all(dots <= 2):
                selected.append(v)
        return selected
    
    def greedy_maxmin(order, time_limit):
        """Greedy with max-min heuristic: pick vector maximizing min distance to selected."""
        selected = []
        remaining = set(order)
        for _ in range(len(order)):
            if time.time() > time_limit:
                break
            if not selected:
                selected.append(candidates[order[0]])
                remaining.discard(order[0])
                continue
            
            best_idx = None
            best_min_dist = -1
            for idx in remaining:
                v = candidates[idx]
                dots = np.dot(selected, v)
                # |v-u|^2 = 8 - 2*v·u, so max min dist = min(8-2*dots)
                min_dist = np.min(8 - 2 * dots)
                if min_dist >= 4 and min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            if best_idx is None:
                break
            selected.append(candidates[best_idx])
            remaining.discard(best_idx)
        
        return selected
    
    # Try multiple strategies
    best_selected = []
    np.random.seed(42)
    
    # Strategy 1: Original fixed order
    fixed_order = list(range(n_candidates))
    sel = greedy_select(fixed_order, deadline - 10.0)
    if len(sel) > len(best_selected):
        best_selected = sel
    
    # Strategy 2: Multiple random orderings with simple greedy
    time_per_trial = max(1.0, (deadline - time.time() - 10.0) / 50.0)
    for trial in range(50):
        if time.time() > deadline - 10.0:
            break
        perm = np.random.permutation(n_candidates)
        sel = greedy_select(perm, deadline - 10.0)
        if len(sel) > len(best_selected):
            best_selected = sel
    
    # Strategy 3: Max-min greedy with random starts
    if time.time() < deadline - 30.0:
        for trial in range(10):
            if time.time() > deadline - 10.0:
                break
            perm = np.random.permutation(n_candidates)
            sel = greedy_maxmin(perm, deadline - 10.0)
            if len(sel) > len(best_selected):
                best_selected = sel
    
    if not best_selected:
        # Fallback: return a minimal valid set
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return np.array(best_selected, dtype=np.int64)
# EVOLVE-BLOCK-END
