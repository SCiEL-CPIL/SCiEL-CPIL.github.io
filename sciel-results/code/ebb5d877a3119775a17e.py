"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _valid_config(vecs, max_norm_sq):
    """Check that all pairwise distances are >= max_norm (squared: >= max_norm_sq)."""
    n = len(vecs)
    if n == 0:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            diff = vecs[i] - vecs[j]
            if np.dot(diff, diff) < max_norm_sq:
                return False
    return True


def _greedy_build(candidates, max_norm_sq, rng):
    """Greedy: pick candidates in random order, keep if compatible with all kept."""
    kept = []
    order = rng.permutation(len(candidates))
    for idx in order:
        v = candidates[idx]
        ok = True
        for u in kept:
            diff = v - u
            if np.dot(diff, diff) < max_norm_sq:
                ok = False
                break
        if ok:
            kept.append(v)
    return kept


def _try_max_norm(candidates, max_norm_sq, rng, time_budget):
    """Try to build a valid set with given max_norm_sq using greedy with restarts."""
    best = []
    t_end = time.time() + time_budget
    while time.time() < t_end:
        res = _greedy_build(candidates, max_norm_sq, rng)
        if len(res) > len(best):
            best = res
            if len(best) >= len(candidates):
                break
    return best


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN

    # Generate candidate vectors: weight-2 (±1 on two coords) and weight-3 (±1 on three coords)
    candidates = []
    
    # Weight-2: choose 2 positions, each ±1
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # Weight-3: choose 3 positions, each ±1
    for i, j, k in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Norms: weight-2 has norm^2=2, weight-3 has norm^2=3
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # We want min pairwise distance >= max norm among selected vectors.
    # If we select only weight-2 vectors, max_norm_sq = 2, need pairwise dist^2 >= 2.
    # If we mix, max_norm_sq could be 3, need pairwise dist^2 >= 3.
    # 
    # Strategy: try different max_norm_sq thresholds and maximize count.
    # For a set S, the constraint is: for all u,v in S, ||u-v||^2 >= max_{w in S} ||w||^2
    # 
    # Let's try: 
    # 1) Only weight-2 vectors, require pairwise dist^2 >= 2 (max norm^2 = 2)
    # 2) Mix weight-2 and weight-3, require pairwise dist^2 >= 3 (max norm^2 = 3)
    # 3) Only weight-3 vectors, require pairwise dist^2 >= 3
    
    best_result = []
    
    # Option 1: weight-2 only, threshold 2
    w2 = candidates[norms_sq == 2]
    rng = np.random.default_rng(42)
    
    t_left = deadline - time.time()
    if t_left > 2.0:
        budget = min(t_left * 0.3, 30.0)
        res = _try_max_norm(w2, 2, rng, budget)
        if len(res) > len(best_result):
            best_result = res
    
    # Option 2: all candidates, threshold 3 (max norm^2 = 3)
    t_left = deadline - time.time()
    if t_left > 2.0:
        budget = min(t_left * 0.4, 40.0)
        res = _try_max_norm(candidates, 3, rng, budget)
        if len(res) > len(best_result):
            best_result = res
    
    # Option 3: weight-3 only, threshold 3
    w3 = candidates[norms_sq == 3]
    t_left = deadline - time.time()
    if t_left > 2.0:
        budget = min(t_left * 0.3, 30.0)
        res = _try_max_norm(w3, 3, rng, budget)
        if len(res) > len(best_result):
            best_result = res
    
    # Try to improve best_result by attempting to add more vectors with lower threshold
    # If best_result contains only weight-2, we can try threshold 2
    if len(best_result) > 0:
        max_ns = max(np.sum(v**2) for v in best_result)
        t_left = deadline - time.time()
        if t_left > 5.0:
            # Try greedy expansion with the actual max norm of current set
            rng2 = np.random.default_rng(123)
            for _ in range(3):
                if time.time() > deadline:
                    break
                # Shuffle candidates not in best_result
                in_set = set(map(tuple, best_result))
                remaining = [c for c in candidates if tuple(c) not in in_set]
                if not remaining:
                    break
                rem_arr = np.array(remaining, dtype=np.int64)
                order = rng2.permutation(len(rem_arr))
                new_best = list(best_result)
                for idx in order:
                    v = rem_arr[idx]
                    # Check against all in new_best
                    ok = True
                    for u in new_best:
                        diff = v - u
                        if np.dot(diff, diff) < max_ns:
                            ok = False
                            break
                    if ok:
                        new_best.append(v)
                if len(new_best) > len(best_result):
                    best_result = new_best
    
    # Final verification
    if len(best_result) == 0:
        # Fallback: return seed-like result
        vecs = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)
    
    result = np.array(best_result, dtype=np.int64)
    
    # Verify validity
    n = len(result)
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            diff = result[i] - result[j]
            if np.dot(diff, diff) < max_norm_sq:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to seed
        vecs = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
