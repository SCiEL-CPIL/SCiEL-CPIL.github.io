import time
import numpy as np
import math
from itertools import combinations

_DEADLINE = 140.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _score(U):
    """Compute the score for a set U containing 0."""
    U = np.asarray(U, dtype=np.int64)
    if len(U) < 2:
        return 1.0
    # difference set
    diff = U[:, None] - U[None, :]
    n_diff = len(np.unique(diff))
    # sumset
    s = U[:, None] + U[None, :]
    n_sum = len(np.unique(s))
    mx = int(U.max())
    denom = math.log2(2 * mx + 1)
    if denom <= 0:
        return 1.0
    ratio = n_diff / n_sum
    if ratio <= 1.0:
        return 1.0
    return 1.0 + math.log2(ratio) / denom

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using known values or greedy construction."""
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99],
        16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115],
        17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132],
        18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151],
        19: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171],
        20: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 194],
    }
    if n in known:
        return np.array(known[n], dtype=np.int64)
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return np.array(ruler, dtype=np.int64)

def _symmetric_core(M, S):
    """
    Construct U = {0} ∪ {x, M-x | x in S} where S is a subset of [1, M-1].
    This exploits symmetry to concentrate sumset around M and 2M while
    spreading difference set across [-M, M].
    """
    U = {0}
    for x in S:
        if 1 <= x <= M - 1:
            U.add(x)
            if M - x != x:  # avoid duplicates when x = M/2
                U.add(M - x)
    return np.array(sorted(U), dtype=np.int64)

def _local_search(U, rng, max_iter=5000, time_budget=15.0):
    """Local search with add/remove/move operations."""
    U = np.sort(np.unique(U.astype(np.int64)))
    cur = _score(U)
    best = U.copy()
    best_score = cur
    start_time = time.time()
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 10 or time.time() - start_time > time_budget:
            break
        
        r = rng.random()
        if r < 0.35:
            # add element
            mx = U.max()
            margin = max(10, mx // 5)
            cand = int(rng.integers(0, mx + margin + 1))
            if cand in U:
                continue
            newU = np.sort(np.append(U, cand))
        elif r < 0.65:
            # remove element (keep 0)
            if len(U) <= 3:
                continue
            idx = int(rng.integers(1, len(U)))
            newU = np.delete(U, idx)
        else:
            # move element
            if len(U) <= 3:
                continue
            idx = int(rng.integers(1, len(U)))
            mx = U.max()
            margin = max(10, mx // 5)
            newval = int(rng.integers(0, mx + margin + 1))
            if newval in U:
                continue
            newU = np.sort(np.delete(np.append(U, newval), idx))
        
        s = _score(newU)
        if s >= cur:
            cur = s
            U = newU
            if cur > best_score:
                best_score = cur
                best = U.copy()
    
    return best, best_score

def _simulated_annealing(U, rng, max_iter=10000, T0=0.1, T_end=0.001, cooling=0.9995, time_budget=20.0):
    """Simulated annealing polish pass."""
    U = np.sort(np.unique(U.astype(np.int64)))
    cur = _score(U)
    best = U.copy()
    best_score = cur
    T = T0
    start_time = time.time()
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 10 or time.time() - start_time > time_budget:
            break
        T *= cooling
        if T < T_end:
            T = T_end
        
        r = rng.random()
        if r < 0.35:
            mx = U.max()
            margin = max(10, mx // 5)
            cand = int(rng.integers(0, mx + margin + 1))
            if cand in U:
                continue
            newU = np.sort(np.append(U, cand))
        elif r < 0.65:
            if len(U) <= 3:
                continue
            idx = int(rng.integers(1, len(U)))
            newU = np.delete(U, idx)
        else:
            if len(U) <= 3:
                continue
            idx = int(rng.integers(1, len(U)))
            mx = U.max()
            margin = max(10, mx // 5)
            newval = int(rng.integers(0, mx + margin + 1))
            if newval in U:
                continue
            newU = np.sort(np.delete(np.append(U, newval), idx))
        
        s = _score(newU)
        delta = s - cur
        
        if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-10)):
            U = newU
            cur = s
            if cur > best_score:
                best_score = cur
                best = U.copy()
    
    return best, best_score

def construct():
    global _START
    _START = time.time()
    rng = np.random.default_rng(42)
    
    best = np.array([0, 1, 3], dtype=np.int64)
    best_score = _score(best)
    
    # Phase 1: Try known Golomb rulers
    for n in range(3, 21):
        if _elapsed() > _DEADLINE - 60:
            break
        try:
            U = _golomb_ruler(n)
            if len(U) < 2 or 0 not in U:
                continue
            s = _score(U)
            if s > best_score:
                best_score = s
                best = U.copy()
        except Exception:
            continue
    
    # Phase 2: Try symmetric core constructions
    for M in range(10, 60, 3):
        if _elapsed() > _DEADLINE - 40:
            break
        # Try small subsets S of [1, M-1]
        max_k = min(5, M // 2)
        for k in range(2, max_k + 1):
            # Sample a few random subsets
            for trial in range(3):
                if _elapsed() > _DEADLINE - 35:
                    break
                if k <= M - 1:
                    S = set(rng.choice(np.arange(1, M), size=k, replace=False))
                else:
                    S = set(range(1, M))
                U = _symmetric_core(M, S)
                if len(U) < 2:
                    continue
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best = U.copy()
    
    # Phase 3: Random seeds
    for trial in range(15):
        if _elapsed() > _DEADLINE - 30:
            break
        n = int(rng.integers(4, 15))
        max_val = int(rng.integers(15, 60))
        U = np.sort(rng.choice(max_val + 1, size=n, replace=False))
        if 0 not in U:
            U[0] = 0
        s = _score(U)
        if s > best_score:
            best_score = s
            best = U.copy()
    
    # Phase 4: Local search on the best solution
    if _elapsed() < _DEADLINE - 25:
        best, best_score = _local_search(best, rng, max_iter=8000, time_budget=20.0)
    
    # Phase 5: Try local search from several diverse seeds
    diverse_seeds = [
        _golomb_ruler(8),
        _golomb_ruler(12),
        np.array([0, 2, 5, 11, 18, 27, 38], dtype=np.int64),
        _symmetric_core(20, {3, 7, 11}),
        _symmetric_core(25, {4, 9, 14}),
    ]
    for seed in diverse_seeds:
        if _elapsed() > _DEADLINE - 15:
            break
        if len(seed) < 2:
            continue
        candidate, candidate_score = _local_search(seed, rng, max_iter=3000, time_budget=8.0)
        if candidate_score > best_score:
            best_score = candidate_score
            best = candidate.copy()
    
    # Phase 6: Simulated annealing polish on the best
    if _elapsed() < _DEADLINE - 10:
        best, best_score = _simulated_annealing(best, rng, max_iter=12000, T0=0.05, T_end=0.0005, cooling=0.9998, time_budget=15.0)
    
    # Phase 7: Final local search polish
    if _elapsed() < _DEADLINE - 5:
        best, best_score = _local_search(best, rng, max_iter=5000, time_budget=10.0)
    
    # Ensure 0 is in the set and return sorted list
    if 0 not in best:
        best = np.sort(np.append(best, 0))
    
    return list(best)
