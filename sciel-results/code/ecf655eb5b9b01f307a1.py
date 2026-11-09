"""Find 10 distinct multiples of divisors [6,10,14,15,21,22,26,33,35,39]
in a minimal-width window [L, L+W], each <= 1e6."""
import time
from math import gcd
from functools import reduce

_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_DEADLINE = 75.0

def _lcm(a, b):
    return a // gcd(a, b) * b

_LCM_ALL = reduce(_lcm, _DIVS)

def _multiples_in_window(L, W, d):
    """Return list of multiples of d in [L, L+W]."""
    if L > _MAXV:
        return []
    hi = min(L + W, _MAXV)
    k0 = (L + d - 1) // d
    k1 = hi // d
    if k0 > k1:
        return []
    return [k * d for k in range(k0, k1 + 1)]

def _try_assign(L, W):
    """Try to assign distinct multiples of each divisor within [L, L+W].
    Returns list of assigned values or None."""
    candidates = []
    for d in _DIVS:
        cands = _multiples_in_window(L, W, d)
        if not cands:
            return None
        candidates.append(cands)
    
    order = sorted(range(len(_DIVS)), key=lambda i: len(candidates[i]))
    assigned = [None] * len(_DIVS)
    used = set()
    
    def backtrack(idx):
        if idx == len(order):
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if backtrack(idx + 1):
                    return True
                used.discard(val)
                assigned[i] = None
        return False
    
    if backtrack(0):
        return [assigned[i] for i in range(len(_DIVS))]
    return None

def _valid_L_for_W(L, W):
    """Check if each divisor has at least one multiple in [L, L+W]."""
    for d in _DIVS:
        k0 = (L + d - 1) // d
        k1 = (L + W) // d
        if k0 > k1:
            return False
    return True

def construct():
    t0 = time.time()
    best_reps = None
    best_W = float('inf')
    
    for W in range(9, 500):
        if time.time() - t0 > _DEADLINE:
            break
        
        found = False
        
        if W >= 38:
            l_candidates = list(range(1, min(_MAXV - W + 1, 30000)))
            import random
            random.seed(W * 7 + 13)
            for _ in range(1000):
                L = random.randint(1, _MAXV - W)
                l_candidates.append(L)
            
            for L in l_candidates:
                if time.time() - t0 > _DEADLINE:
                    break
                reps = _try_assign(L, W)
                if reps is not None:
                    best_reps = reps
                    best_W = W
                    found = True
                    break
        else:
            l_limit = min(_MAXV - W + 1, 80000)
            for L in range(1, l_limit + 1):
                if time.time() - t0 > _DEADLINE:
                    break
                if not _valid_L_for_W(L, W):
                    continue
                reps = _try_assign(L, W)
                if reps is not None:
                    best_reps = reps
                    best_W = W
                    found = True
                    break
        
        if found:
            break
    
    if best_reps is None:
        import random
        random.seed(42)
        for _ in range(300000):
            if time.time() - t0 > _DEADLINE:
                break
            W = random.randint(30, 800)
            L = random.randint(1, _MAXV - W)
            if not _valid_L_for_W(L, W):
                continue
            reps = _try_assign(L, W)
            if reps is not None:
                if W < best_W:
                    best_reps = reps
                    best_W = W
                    break
    
    if best_reps is None:
        best_reps = list(_DIVS)
    
    return [int(x) for x in best_reps]
