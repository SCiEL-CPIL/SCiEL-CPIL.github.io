import numpy as np
import time
from scipy.optimize import linprog

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]


def construct():
    """
    Solve the LP:
        minimize  c^T x  where c = max_j (autoconvolution at lag j)
        subject to: sum(x) = 1, x >= 0
    
    The autoconvolution at lag j is: sum_i h[i] * h[i+j] (for j >= 0)
    This is quadratic in h, NOT linear.
    
    Wait - the autoconvolution is NOT linear in h. The directive says "minimizing the peak autoconvolution, which is a linear function of the height vector" but that's incorrect - autoconvolution is quadratic.
    
    However, we can reformulate: for a fixed set of non-zero indices (support), if we use a Sidon set, the autoconvolution peak is minimized when weights are uniform (or we can optimize weights).
    
    Actually, let me reconsider. The autoconvolution (h*h)[k] = sum_i h[i]*h[i+k]. This is quadratic.
    
    But we can use the fact that for a Sidon set S of size K, all pairwise sums i+j are distinct (for i <= j in S), so the autoconvolution has at most K(K+1)/2 non-zero entries, each being a product of two weights.
    
    For a Sidon set with uniform weights w = 1/K, the autoconvolution peak is w^2 = 1/K^2 (since each sum appears at most once).
    
    The score is: L2^2 / (L1 * peak_autoconv * DELTA)
    With L1 = 1: L2^2 = K * w^2 = K/K^2 = 1/K
    peak_autoconv = w^2 = 1/K^2
    Score = (1/K) / (1 * (1/K^2) * DELTA) = K / DELTA = K * N * 2 = 2*N*K
    
    So to maximize score, we want to maximize K, the size of the Sidon set.
    
    For N=50, the maximum Sidon set size is known to be around 8-9.
    
    Let me find the largest Sidon set and use uniform weights.
    """
    
    def is_sidon(indices):
        """Check if indices form a Sidon set."""
        sums = set()
        n = len(indices)
        for i in range(n):
            for j in range(i, n):
                s = indices[i] + indices[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def find_max_sidon(n, max_k=10):
        """Find a Sidon set of maximum size using backtracking."""
        
        def backtrack(start, current, used_sums):
            # Try to extend
            best = current[:]
            for i in range(start, n):
                # Check if i can be added
                valid = True
                for s in current:
                    if (s + i) in used_sums:
                        valid = False
                        break
                if not valid:
                    continue
                if (i + i) in used_sums:
                    continue
                
                new_sums = set()
                for s in current:
                    new_sums.add(s + i)
                new_sums.add(i + i)
                
                current.append(i)
                used_sums.update(new_sums)
                
                result = backtrack(i + 1, current, used_sums)
                if len(result) > len(best):
                    best = result[:]
                
                used_sums.difference_update(new_sums)
                current.pop()
            
            return best
        
        return backtrack(0, [], set())
    
    # Find maximum Sidon set
    sidon = find_max_sidon(N, 10)
    K = len(sidon)
    
    if K < 2:
        # Fallback
        h = np.ones(N) / N
        return list(h)
    
    # Use uniform weights on the Sidon set
    h = np.zeros(N)
    for idx in sidon:
        h[idx] = 1.0 / K
    
    # Try to improve with non-uniform weights using a local optimization
    # The score is L2^2 / (L1 * peak * DELTA)
    # With L1=1: score = sum(h^2) / (peak_autoconv * DELTA)
    # For Sidon set, peak_autoconv = max over all sums of h[i]*h[j] where i+j=sum
    # Each sum appears at most once, so peak = max_{i<=j in S} h[i]*h[j]
    
    # We want to maximize sum(h^2) / max_{i<=j} h[i]*h[j]
    # This is a ratio optimization. Let's try a few heuristic weightings.
    
    best_score = 0.0
    best_h = h.copy()
    
    # Try uniform
    def compute_score(hv):
        hv = np.asarray(hv, dtype=np.float64)
        l1 = np.sum(hv)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(hv**2)
        h_conv = np.convolve(hv, hv, mode='full')
        autoconv_max = DELTA * np.max(h_conv)
        if autoconv_max == 0:
            return 0.0
        return l2_sq / (l1 * autoconv_max)
    
    best_score = compute_score(h)
    
    # Try geometric weighting
    for base in [0.8, 0.9, 1.1, 1.2]:
        hw = np.zeros(N)
        for k, idx in enumerate(sidon):
            hw[idx] = base ** k
        hw = hw / np.sum(hw)
        s = compute_score(hw)
        if s > best_score:
            best_score = s
            best_h = hw.copy()
    
    # Try random weightings
    rng = np.random.default_rng(42)
    for _ in range(200):
        hw = rng.random(K)
        hw = hw / np.sum(hw)
        hv = np.zeros(N)
        for k, idx in enumerate(sidon):
            hv[idx] = hw[k]
        s = compute_score(hv)
        if s > best_score:
            best_score = s
            best_h = hv.copy()
    
    # Try to optimize with scipy on the Sidon support
    from scipy.optimize import minimize
    
    def obj(x):
        hv = np.zeros(N)
        for k, idx in enumerate(sidon):
            hv[idx] = x[k]
        # score = L2^2 / (L1 * peak * DELTA), with L1=1
        l2_sq = np.sum(hv**2)
        h_conv = np.convolve(hv, hv, mode='full')
        peak = DELTA * np.max(h_conv)
        if peak == 0:
            return 0.0
        return -l2_sq / peak  # negate for minimization
    
    x0 = np.ones(K) / K
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    bounds = [(0, None)] * K
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-12})
        if res.success:
            hv = np.zeros(N)
            for k, idx in enumerate(sidon):
                hv[idx] = max(0, res.x[k])
            hv = hv / np.sum(hv)
            s = compute_score(hv)
            if s > best_score:
                best_score = s
                best_h = hv.copy()
    except Exception:
        pass
    
    return list(best_h)
