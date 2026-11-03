import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA
_DEADLINE = 135.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _ratio(h):
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return -np.inf
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 1e-15:
        return 0.0
    return l2_sq / (l1 * m)


def _find_sidon_sets(k, n_tries=200):
    """Find Sidon sets of size k in [0, N) with unique pairwise sums."""
    rng = np.random.RandomState(k * 7919 + 13)
    best_sets = []
    best_min_gap = -1
    
    for _ in range(n_tries):
        S = []
        sums = set()
        success = True
        
        for _ in range(k):
            found = False
            for _trial in range(200):
                cand = rng.randint(0, N)
                if cand in S:
                    continue
                
                new_sums = set()
                conflict = False
                for s in S:
                    ps = cand + s
                    if ps in sums:
                        conflict = True
                        break
                    new_sums.add(ps)
                if not conflict and cand + cand not in sums:
                    new_sums.add(cand + cand)
                    S.append(cand)
                    sums.update(new_sums)
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success and len(S) == k:
            S_sorted = sorted(S)
            gaps = np.diff(S_sorted)
            min_gap = np.min(gaps) if len(gaps) > 0 else 0
            if min_gap > best_min_gap:
                best_min_gap = min_gap
                best_sets = [S_sorted]
            elif min_gap == best_min_gap:
                best_sets.append(S_sorted)
    
    # Also try structured sets
    structured = []
    # Powers of 2
    pw2 = [1 << i for i in range(k) if (1 << i) < N]
    if len(pw2) == k:
        structured.append(pw2)
    
    # Consecutive primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if len(primes) >= k:
        structured.append(primes[:k])
    
    # Arithmetic progressions with large step
    for start in range(0, 5):
        for step in range(2, N // k + 1):
            S = [start + i * step for i in range(k)]
            if all(s < N for s in S):
                # Check Sidon property
                sums = set()
                valid = True
                for i in range(k):
                    for j in range(i, k):
                        s = S[i] + S[j]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    structured.append(S)
    
    all_sets = best_sets + structured
    # Deduplicate
    unique = []
    seen = set()
    for S in all_sets:
        key = tuple(sorted(S))
        if key not in seen:
            seen.add(key)
            unique.append(list(S))
    
    return unique[:20]  # Return top 20


def _optimize_for_support(S, h_init=None, maxiter=500):
    """Optimize weights on a given support using SLSQP with M variable."""
    n = len(S)
    
    if h_init is None:
        h_vals = np.ones(n) / n
    else:
        h_vals = np.array([h_init[i] for i in S], dtype=np.float64)
        s = np.sum(h_vals)
        if s > 0:
            h_vals = h_vals / s
        else:
            h_vals = np.ones(n) / n
    
    M_init = np.max(np.convolve(h_vals, h_vals))
    
    def obj(x):
        return x[n]  # Minimize M
    
    def obj_grad(x):
        g = np.zeros(n + 1)
        g[n] = 1.0
        return g
    
    def con_sum(x):
        return np.sum(x[:n]) - 1.0
    
    def con_sum_grad(x):
        g = np.zeros(n + 1)
        g[:n] = 1.0
        return g
    
    def con_auto(x):
        hv = x[:n]
        M = x[n]
        c = np.convolve(hv, hv)
        return M - c
    
    def con_auto_grad(x):
        hv = x[:n]
        c_len = 2 * n - 1
        grad = np.zeros((c_len, n + 1))
        for k in range(c_len):
            derivs = np.zeros(n)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    derivs[i] = 2.0 * hv[j]
            grad[k, :n] = -derivs
            grad[k, n] = 1.0
        return grad
    
    bounds = [(0.0, None)] * n + [(0.0, None)]
    cons = [
        {'type': 'eq', 'fun': con_sum, 'jac': con_sum_grad},
        {'type': 'ineq', 'fun': con_auto, 'jac': con_auto_grad}
    ]
    
    x0 = np.concatenate([h_vals, [M_init]])
    
    try:
        res = minimize(obj, x0, method='SLSQP', jac=obj_grad,
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-10})
        
        if res.x is not None:
            h_opt = res.x[:n]
            h_opt = np.clip(h_opt, 0.0, None)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                # Map back to full vector
                h_full = np.zeros(N)
                for i, idx in enumerate(S):
                    h_full[idx] = h_opt[i]
                return h_full
    except Exception:
        pass
    
    return None


def _local_polish(h, maxiter=300):
    """Local refinement on the full N-dimensional space."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if s <= 0:
        return h
    
    h = h / s
    
    def neg_log_ratio(x):
        x = np.maximum(x, 0.0)
        l1 = np.sum(x)
        if l1 <= 1e-15:
            return 1e10
        l2_sq = np.dot(x, x)
        if l2_sq <= 1e-15:
            return 1e10
        conv = np.convolve(x, x)
        m = np.max(conv)
        if m <= 1e-15:
            return 1e10
        return -np.log(l2_sq) + np.log(l1) + np.log(m)
    
    def neg_log_ratio_grad(x):
        x = np.maximum(x, 0.0)
        l1 = np.sum(x)
        l2_sq = np.dot(x, x)
        if l1 <= 1e-15 or l2_sq <= 1e-15:
            return 1e10, np.zeros(N)
        
        conv = np.convolve(x, x)
        m = np.max(conv)
        if m <= 1e-15:
            return 1e10, np.zeros(N)
        
        val = -np.log(l2_sq) + np.log(l1) + np.log(m)
        
        grad = -2.0 * x / l2_sq
        grad += 1.0 / l1
        
        idx_max = np.argmax(conv)
        g = np.zeros(N)
        k = idx_max
        for j in range(N):
            kj = k - j
            if 0 <= kj < N:
                g[j] = 2.0 * x[kj]
        grad += (1.0 / m) * g
        
        return val, grad
    
    bounds = [(0.0, 1.0)] * N
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
             'jac': lambda x: np.ones(N)}]
    
    try:
        res = minimize(neg_log_ratio, h, method='SLSQP',
                       jac=neg_log_ratio_grad, bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.x is not None:
            hx = np.clip(res.x, 0.0, None)
            s2 = np.sum(hx)
            if s2 > 1e-15:
                hx = hx / s2
                if _ratio(hx) > _ratio(h):
                    return hx
    except Exception:
        pass
    
    return h


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
    
    # Phase 1: Sidon set search and optimization
    all_sidon_sets = []
    for k in range(4, 13):
        if _time_left() < 15:
            break
        sets = _find_sidon_sets(k, n_tries=100)
        all_sidon_sets.extend(sets)
    
    # Phase 2: Optimize weights for each Sidon set
    for S in all_sidon_sets:
        if _time_left() < 10:
            break
        
        # Try multiple initializations
        inits = [None]
        
        # Geometric initializations
        for r in [0.5, 0.8, 1.2, 2.0]:
            h_g = np.zeros(N)
            for i, idx in enumerate(S):
                h_g[idx] = r ** i
            inits.append(h_g)
        
        # Random initializations
        for _ in range(3):
            h_r = np.zeros(N)
            for idx in S:
                h_r[idx] = np.random.uniform(0.1, 1.0)
            inits.append(h_r)
        
        for h_init in inits:
            if _time_left() < 5:
                break
            h_opt = _optimize_for_support(S, h_init, maxiter=300)
            if h_opt is not None:
                upd(h_opt)
    
    # Phase 3: Diverse random initializations with local polish
    x_idx = np.arange(N)
    
    # Sparse random
    for _ in range(15):
        if _time_left() < 10:
            break
        k = np.random.randint(4, 14)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = np.random.uniform(0.1, 1.0, k)
        upd(_local_polish(h, maxiter=200))
    
    # Gaussian bumps
    for c in np.linspace(3, N-3, 8):
        if _time_left() < 10:
            break
        for w in [1.5, 2.5, 4.0, 6.0]:
            h = np.exp(-0.5 * ((x_idx - c) / w) ** 2)
            upd(_local_polish(h, maxiter=200))
    
    # Triangular
    for c in np.linspace(3, N-3, 6):
        if _time_left() < 10:
            break
        for w in [3, 5, 8, 12]:
            h = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            upd(_local_polish(h, maxiter=200))
    
    # Peaky: one large, rest small
    for _ in range(8):
        if _time_left() < 8:
            break
        h = np.random.rand(N) * 0.01
        h[np.random.randint(N)] = 1.0
        upd(_local_polish(h, maxiter=200))
    
    # Phase 4: Final tight polish
    if _time_left() > 10:
        upd(_local_polish(best_h, maxiter=1000))
    
    return list(best_h)
