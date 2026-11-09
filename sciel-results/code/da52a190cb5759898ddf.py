N = 16
import time
import random
import numpy as np

# Precompute grid and distances
_ALL = [(i, j) for i in range(N) for j in range(N)]
_M = N * N
# D[i, j] = squared distance between point i and point j
D = np.zeros((_M, _M), dtype=np.int64)
for i in range(_M):
    x1, y1 = _ALL[i]
    for j in range(_M):
        x2, y2 = _ALL[j]
        D[i, j] = (x1 - x2) ** 2 + (y1 - y2) ** 2

DEADLINE = time.time() + 280.0

def _can_add(p, chosen):
    """
    p: int
    chosen: list of ints
    """
    if len(chosen) < 2:
        return True
    
    # Get distances from p to all chosen points
    d_p = D[p, chosen]
    
    # Check for duplicate distances from p (isosceles with p as apex)
    if len(np.unique(d_p)) < len(d_p):
        return False
    
    # Check for isosceles with base in chosen
    # For each pair (i, j) in chosen, check if d_p[i] == D[i, j] or d_p[j] == D[i, j]
    # d_p[k] is distance from p to chosen[k]
    # D[chosen[i], chosen[j]] is distance between chosen[i] and chosen[j]
    
    K = len(chosen)
    # Vectorized check:
    # We need to check if any d_p[i] is in D[chosen[i], chosen] excluding i
    # This is tricky to fully vectorize simply without loops, but K is small.
    
    for i in range(K):
        # d_pi is distance from p to chosen[i]
        d_pi = d_p[i]
        # Get distances from chosen[i] to all other chosen points
        d_ij = D[chosen[i], chosen]
        # Mask out i itself
        d_ij[i] = -1 # or large number, since distances are >= 0
        
        # Check if d_pi is in d_ij
        if d_pi in d_ij:
            return False
            
    return True

def _greedy(order):
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen

def _is_maximal(chosen):
    chosen_set = set(chosen)
    for p in range(_M):
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True

def _local_improve(chosen, deadline):
    chosen = [int(c) for c in chosen] # Ensure Python ints
    improved = True
    while improved and time.time() < deadline:
        improved = False
        chosen_set = set(chosen)
        for ri, r in enumerate(list(chosen)):
            base = [c for c in chosen if c != r]
            base_set = set(base)
            
            # Find candidates that can be added to base
            # Optimization: only check points not in chosen
            cand = []
            for p in range(_M):
                if p in chosen_set:
                    continue
                if _can_add(p, base):
                    cand.append(p)
            
            if len(cand) < 2:
                continue
                
            # Try to find two candidates p1, p2 such that p2 can be added to base + [p1]
            found = None
            for a in range(len(cand)):
                p1 = cand[a]
                # Check if p1 can be added to base (already known true)
                # Check if any other cand can be added to base + [p1]
                for b in range(a + 1, len(cand)):
                    p2 = cand[b]
                    if _can_add(p2, base + [p1]):
                        found = (p1, p2)
                        break
                if found:
                    break
                    
            if found:
                new_chosen = base + list(found)
                # Verify
                if _is_maximal(new_chosen):
                    chosen = new_chosen
                    improved = True
                    break
    return chosen

def construct():
    random.seed(12345)
    best = None
    best_len = float('inf')
    
    # Initial greedy
    order0 = list(range(_M))
    c0 = _greedy(order0)
    best = c0
    best_len = len(c0)
    
    while time.time() < DEADLINE:
        order = list(range(_M))
        random.shuffle(order)
        c = _greedy(order)
        c = _local_improve(c, DEADLINE)
        
        if len(c) < best_len:
            best = c
            best_len = len(c)
            
        if best_len <= 14:
            break
            
    return [(int(_ALL[i][0]), int(_ALL[i][1])) for i in best]
