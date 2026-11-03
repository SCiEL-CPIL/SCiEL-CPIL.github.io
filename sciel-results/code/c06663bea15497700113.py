import numpy as np
import time
import sys
from scipy.optimize import minimize

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 140.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv). Returns -inf/0 if degenerate."""
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


def _neg_log_ratio(h):
    """
    Minimize -log(ratio) = -log(L2^2) + log(L1) + log(M).
    Returns (value, gradient).
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    l2_sq = np.dot(h, h)
    if l1 <= 0 or l2_sq <= 0:
        return 0.0, np.zeros(N)
    
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0:
        return 0.0, np.zeros(N)
    
    val = -np.log(l2_sq) + np.log(l1) + np.log(m)
    
    # Gradient
    grad = -2.0 * h / l2_sq
    grad += 1.0 / l1
    
    idx_max = np.argmax(conv)
    g = np.zeros(N)
    k = idx_max
    for j in range(N):
        kj = k - j
        if 0 <= kj < N:
            g[j] = 2.0 * h[kj]
            
    grad += (1.0 / m) * g
    
    return val, grad


def _polish(h, maxiter=500):
    """Local refinement maximizing ratio under L1=1, h>=0 using analytic gradient."""
    h0 = h / np.sum(h)
    bounds = [(0.0, 1.0)] * N
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
             'jac': lambda x: np.ones(N)}]
    try:
        res = minimize(_neg_log_ratio, h0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False})
        if res.x is not None:
            hx = np.clip(res.x, 0.0, None)
            if np.sum(hx) > 0:
                hx = hx / np.sum(hx)
                if _ratio(hx) > _ratio(h0):
                    return hx
    except Exception:
        pass
    return h0


def _find_sidon_set(k, seed=0):
    """
    Find a Sidon set of size k in [0, N).
    Uses a greedy approach with randomization.
    """
    rng = np.random.RandomState(seed)
    best_set = None
    best_score = -1
    
    for attempt in range(100):
        S = []
        sums = set()
        
        # Randomize order of candidates to avoid bias
        candidates = list(range(N))
        rng.shuffle(candidates)
        
        for cand in candidates:
            if len(S) == k:
                break
            
            # Check if adding cand creates a collision
            conflict = False
            new_sums = set()
            for s in S:
                pair_sum = cand + s
                if pair_sum in sums:
                    conflict = True
                    break
                new_sums.add(pair_sum)
            
            if not conflict:
                if cand + cand in sums:
                    conflict = True
                else:
                    new_sums.add(cand + cand)
            
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        
        if len(S) == k:
            # Score: prefer sets with larger minimum gap
            S_sorted = sorted(S)
            gaps = np.diff(S_sorted)
            score = np.min(gaps) if len(gaps) > 0 else 0
            if score > best_score:
                best_score = score
                best_set = S_sorted
    
    if best_set is None:
        # Fallback: random set
        best_set = sorted(rng.choice(N, k, replace=False).tolist())
    
    return best_set


def construct():
    np.random.seed(12345)
    best_h = np.ones(N) / N
    best_r = _ratio(best_h)

    def upd(h):
        nonlocal best_h, best_r
        h = np.asarray(h, dtype=np.float64)
        if np.sum(h) <= 0 or not np.all(np.isfinite(h)):
            return
        hh = np.clip(h, 0.0, None)
        r = _ratio(hh)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_h = hh.copy()

    # ---------------- PHASE 1: Sidon Set Search & Weight Optimization ----------------
    candidate_supports = []
    
    # 1. Small Sidon sets (sparse)
    for k in range(3, 15):
        for seed in range(10):
            S = _find_sidon_set(k, seed=seed + k*100)
            candidate_supports.append(S)
    
    # 2. Specific known Sidon sets
    candidate_supports.append([0, 1, 2, 4, 8, 16, 32])
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    candidate_supports.append([p for p in primes if p < N])
    
    # 3. Random smooth sets
    for _ in range(10):
        a = np.random.uniform(0.5, 5.0)
        b = np.random.uniform(-0.9, 0.9)
        c = np.random.uniform(4, N - 4)
        h_bump = np.exp(-a * (np.arange(N) - c) ** 2) * (1.0 + b * np.cos(0.5 * np.pi * np.arange(N) / N))
        threshold = np.percentile(h_bump, 50)
        support = np.where(h_bump > threshold)[0].tolist()
        if len(support) > 3:
            candidate_supports.append(support)

    # Optimize weights for each support
    for S in candidate_supports:
        if _time_left() < 10:
            break
        
        # Create initial vector with ones on support
        h_init = np.zeros(N)
        for idx in S:
            h_init[idx] = 1.0
        
        # Try different initial weight distributions
        h_cands = []
        h_cands.append(h_init.copy())
        
        # Geometric
        for r in [0.5, 0.8, 1.2, 2.0]:
            h_g = np.zeros(N)
            for i, idx in enumerate(S):
                h_g[idx] = r ** i
            h_cands.append(h_g)
        
        # Random
        for _ in range(5):
            h_r = np.zeros(N)
            for idx in S:
                h_r[idx] = np.random.uniform(0.1, 1.0)
            h_cands.append(h_r)
        
        for h_c in h_cands:
            if _time_left() < 5:
                break
            # Normalize
            h_c = h_c / np.sum(h_c)
            # Polish
            h_opt = _polish(h_c, maxiter=200)
            upd(h_opt)

    # ---------------- PHASE 2: Global Random Search with Polish ----------------
    inits = [best_h.copy(), np.ones(N) / N]
    
    # More random smooth starts
    for _ in range(10):
        if _time_left() < 15:
            break
        a = np.random.uniform(0.5, 8.0)
        b = np.random.uniform(-0.9, 0.9)
        c = np.random.uniform(2, N - 2)
        inits.append(np.exp(-a * (np.arange(N) - c) ** 2) * (1.0 + b * np.cos(0.5 * np.pi * np.arange(N) / N)) / N)
    
    # Sparse random starts
    for _ in range(10):
        if _time_left() < 15:
            break
        k = np.random.randint(4, 15)
        h = np.zeros(N)
        h[np.random.choice(N, k, replace=False)] = np.random.uniform(0.1, 1.0, k)
        inits.append(h)
    
    # "Peaky" starts: one large, rest small
    for _ in range(5):
        if _time_left() < 15:
            break
        h = np.random.rand(N) * 0.01
        h[np.random.randint(N)] = 1.0
        inits.append(h)

    for h in inits:
        if _time_left() < 12:
            break
        h = np.clip(h, 0.0, None)
        if np.sum(h) <= 0:
            continue
        upd(_polish(h / np.sum(h), maxiter=250))

    # ---------------- PHASE 3: Final Tight Polish ----------------
    if _time_left() > 8:
        upd(_polish(best_h, maxiter=1500))

    return list(best_h)
