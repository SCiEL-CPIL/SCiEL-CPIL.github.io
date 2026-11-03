import numpy as np
import time

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 140.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return -np.inf
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)

def _find_max_sidon_set(N, max_k=20, iterations=500):
    """
    Find the largest Sidon set in [0, N) using randomized greedy with restarts.
    A Sidon set S satisfies that all sums i+j for i,j in S are unique.
    """
    best_set = []
    rng = np.random.RandomState(42)
    
    for _ in range(iterations):
        S = []
        sums = set()
        candidates = list(range(N))
        rng.shuffle(candidates)
        
        for c in candidates:
            if len(S) >= max_k:
                break
            conflict = False
            for s in S:
                if c + s in sums:
                    conflict = True
                    break
            if not conflict:
                if c + c in sums:
                    conflict = True
            if not conflict:
                S.append(c)
                for s in S:
                    sums.add(c + s)
        
        if len(S) > len(best_set):
            best_set = S[:]
            if len(best_set) >= max_k:
                break
                
    return best_set

def construct():
    np.random.seed(12345)
    
    # Find the largest Sidon set
    S = _find_max_sidon_set(N, max_k=15, iterations=500)
    
    if not S:
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
    
    k = len(S)
    h = np.zeros(N)
    for idx in S:
        h[idx] = 1.0 / k
        
    return list(h)
