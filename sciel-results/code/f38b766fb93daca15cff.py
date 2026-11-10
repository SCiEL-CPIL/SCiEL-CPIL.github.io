import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix

_DEADLINE = 600.0  # safety margin under 630
_START = time.time()

def _elapsed():
    return time.time() - _START

def _verify_fast(domain, values, K):
    """Fast feasibility check using O(n*sqrt(K)) structure of floor(x/k)."""
    vals = np.clip(values, -10, 10)
    d = domain.astype(float)
    # Adjust f(1)
    S = np.sum(vals / d)
    if 1 in domain:
        idx1 = np.where(domain == 1)[0][0]
        vals = vals.copy()
        vals[idx1] = vals[idx1] - S
    else:
        vals = np.concatenate([vals, [-S]])
        domain = np.concatenate([domain, [1]])
        d = domain.astype(float)
    
    n = len(domain)
    maxK = 10 * K
    
    # For each k, find x ranges where floor(x/k) is constant
    # For x in [1, maxK], floor(x/k) = j for x in [j*k, (j+1)*k - 1]
    # M(x) = sum_k vals[k] * floor(x/k)
    # We need M(x) <= 1.0001 for all x in [1, maxK]
    # 
    # Key insight: M(x) is a step function. It only changes at multiples of k.
    # So we only need to check x values that are multiples of some k, or just after.
    # Actually, M(x) is constant on intervals [j*k, (j+1)*k - 1] for each k.
    # The maximum of M(x) occurs at points where many floors change.
    # 
    # More efficient: M(x) = sum_k vals[k] * floor(x/k)
    # For a fixed x, floor(x/k) = 0 for k > x, and floor(x/k) <= x/k for all k.
    # So M(x) = sum_{k<=x} vals[k] * floor(x/k)
    # 
    # The critical points are x values that are multiples of many small k.
    # Let's check all x that are multiples of any k in domain, plus x=1.
    # But that could be too many. Instead, let's use the fact that:
    # M(x) - M(x-1) = sum_k vals[k] * (floor(x/k) - floor((x-1)/k))
    # floor(x/k) - floor((x-1)/k) = 1 if k|x, else 0
    # So M(x) - M(x-1) = sum_{k|x, k in domain} vals[k]
    # 
    # This means M(x) is a random walk where at each step x, we add the sum of vals[k] for k dividing x.
    # We can compute M(x) for all x in [1, maxK] in O(maxK * log(maxK)) using a sieve-like approach.
    
    M = np.zeros(maxK + 1)
    # For each k in domain, add vals[k] to all multiples of k
    for i in range(n):
        k = int(domain[i])
        v = vals[i]
        if v == 0:
            continue
        # Add v to M[j*k] for j = 1, 2, ... while j*k <= maxK
        for j in range(1, maxK // k + 1):
            M[j * k] += v
    
    # M(x) = sum of contributions from all k dividing x... wait, that's not right.
    # M(x) = sum_k vals[k] * floor(x/k)
    # This is NOT the same as sum of vals[k] for k|x.
    # Let me reconsider.
    # 
    # Actually, the difference M(x) - M(x-1) = sum_{k|x} vals[k] is correct.
    # So M(x) = M(x-1) + sum_{k|x, k in domain} vals[k]
    # M(0) = 0
    # 
    # So we can compute M(x) for all x by:
    # 1. Initialize M = zeros(maxK+1)
    # 2. For each k in domain, for each multiple j*k of k, add vals[k] to a "delta" array
    # 3. Then M(x) = cumsum(delta[1:x+1])
    
    delta = np.zeros(maxK + 1)
    for i in range(n):
        k = int(domain[i])
        v = vals[i]
        if v == 0:
            continue
        for j in range(1, maxK // k + 1):
            delta[j * k] += v
    
    M = np.cumsum(delta)  # M[0]=0, M[x] = sum_{i=1}^{x} delta[i]
    
    return np.all(M[1:] <= 1.0001)

def _compute_a(domain, values):
    vals = np.clip(values, -10, 10)
    d = domain.astype(float)
    S = np.sum(vals / d)
    if 1 in domain:
        idx1 = np.where(domain == 1)[0][0]
        vals = vals.copy()
        vals[idx1] = vals[idx1] - S
    else:
        vals = np.concatenate([vals, [-S]])
        domain = np.concatenate([domain, [1]])
        d = domain.astype(float)
    a = -np.sum(vals * np.log(d) / d)
    return a

def _polish(domain, values, K, max_iters=500):
    """Nelder-Mead polish to improve the objective while maintaining feasibility."""
    def objective(x):
        """x is the vector of f(2)..f(n), we compute the actual objective."""
        d = domain.astype(float)
        # x corresponds to f(2)..f(n) where n = len(domain)
        vals = np.empty(len(d))
        vals[0] = 0.0  # placeholder for f(1)
        vals[1:] = x
        vals = np.clip(vals, -10, 10)
        S = np.sum(vals / d)
        vals[0] = vals[0] - S  # adjust f(1)
        # Check f(1) is within bounds
        if abs(vals[0]) > 10:
            return 1e6
        # Quick feasibility check
        if not _verify_fast(domain, vals, K):
            return 1e6
        a = -np.sum(vals * np.log(d) / d)
        return -a  # minimize -a to maximize a
    
    n = len(domain)
    x0 = values[1:].copy()  # start from current f(2)..f(n)
    
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-10})
    
    if res.success or res.fun < -_compute_a(domain, values):
        new_vals = np.empty(len(domain))
        new_vals[0] = 0.0
        new_vals[1:] = res.x
        # Verify
        if _verify_fast(domain, new_vals, K):
            return domain, new_vals
    return domain, values

def prime_number_theorem_partial_function():
    global _START
    _START = time.time()
    
    # fallback seed
    seed_domain = np.array([1,2,3,5,30], dtype=np.int64)
    seed_values = np.array([3.47155351603266,-1.0,-1.0,-1.0,1.0], dtype=np.float64)
    best = (seed_domain, seed_values)
    best_a = _compute_a(seed_domain, seed_values)
    
    for K in [400, 600, 800, 1200, 1600, 2000]:
        if _elapsed() > _DEADLINE - 180:
            break
        try:
            # variables f(2)..f(K)
            ks = np.arange(2, K+1, dtype=float)
            c = np.log(ks) / ks  # minimize c.f
            n = len(ks)
            # bounds
            bounds = [(-10.0, 10.0)] * n
            # constraints A f <= 1.0001 for x in 1..10K
            maxK = 10 * K
            # Build sparse A: rows = x, cols = k-2
            # A[x-1, k-2] = floor(x/k)
            rows = []
            cols = []
            data = []
            xchunk = 500
            for x0 in range(1, maxK+1, xchunk):
                x1 = min(x0+xchunk, maxK+1)
                xs = np.arange(x0, x1, dtype=float)
                mat = np.floor(xs[:, None] / ks[None, :])
                r = np.repeat(np.arange(x0-1, x1-1), n)
                cc = np.tile(np.arange(n), len(xs))
                rows.append(r); cols.append(cc); data.append(mat.ravel())
            A = csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(maxK, n))
            b = np.full(maxK, 1.0001)
            # f(1) bounds: -10 <= -sum f/k <= 10
            invk = 1.0 / ks
            A1 = csr_matrix((invk, (np.zeros(n), np.arange(n))), shape=(1, n))
            A2 = csr_matrix((-invk, (np.zeros(n), np.arange(n))), shape=(1, n))
            A_all = csr_matrix.vstack([A, A1, A2], format='csr')
            b_all = np.concatenate([b, [10.0], [10.0]])
            res = linprog(c, A_ub=A_all, b_ub=b_all, bounds=bounds, method='highs')
            if res.success:
                f = res.x
                dom = np.arange(2, K+1, dtype=np.int64)
                vals = f
                domain = np.concatenate([[1], dom])
                values = np.concatenate([[0.0], vals])
                # verify with fast method
                if _verify_fast(domain, values, K):
                    a = _compute_a(domain, values)
                    if a > best_a:
                        best_a = a
                        best = (domain, values)
                    # Try to polish
                    if _elapsed() < _DEADLINE - 60:
                        polished = _polish(domain, values, K, max_iters=200)
                        if polished is not None:
                            a_pol = _compute_a(polished[0], polished[1])
                            if a_pol > best_a:
                                best_a = a_pol
                                best = polished
            if _elapsed() > _DEADLINE - 120:
                break
        except Exception:
            pass
    
    # Final polish on the best solution
    if _elapsed() < _DEADLINE - 30:
        try:
            K_best = int(best[0].max())
            polished = _polish(best[0], best[1], K_best, max_iters=1000)
            if polished is not None:
                a_pol = _compute_a(polished[0], polished[1])
                if a_pol > best_a:
                    best_a = a_pol
                    best = polished
        except Exception:
            pass
    
    return best
