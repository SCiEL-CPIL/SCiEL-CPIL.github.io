# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly for discrete measure."""
    n = len(support)
    if n == 0:
        return 0.0
    w = weights / weights.sum()
    s = support
    
    # For each pair (k, l), we need the probability that s_i + s_j < 2*s_l - s_k
    # P = sum_{k,l} w_k * w_l * P[s_i + s_j < 2*s_l - s_k]
    # where the inner probability is over i,j
    
    # Efficient: for each threshold t, precompute P[s_i + s_j < t]
    # But thresholds depend on k,l, so we compute per (k,l)
    
    total = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Count pairs (i,j) with s_i + s_j < threshold
            # We can use a sorted approach, but for small n, direct is fine
            # For larger n, use convolution or sorting
            if n <= 200:
                # Direct O(n^2) for each (k,l) pair -> O(n^4) total
                # This is too slow for n > 50 or so
                pass
            
            # Use sorting for efficiency: sort s, then for each i, count j with s_j < threshold - s_i
            # But we need weighted sum
            # Let's use a different approach: precompute all pairwise sums
            if n <= 300:
                # Precompute all pairwise weighted sums
                # This is O(n^2) memory
                pass
    
    # Actually, let's just use the direct O(n^4) for small n and a smarter approach for larger
    if n <= 150:
        total = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * s[l] - s[k]
                inner = 0.0
                for i in range(n):
                    for j in range(n):
                        if s[i] + s[j] < threshold:
                            inner += w[i] * w[j]
                total += w[k] * w[l] * inner
        return total
    else:
        # For larger n, use the sorted approach
        # Sort support points
        order = np.argsort(s)
        s_sorted = s[order]
        w_sorted = w[order]
        
        total = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * s[l] - s[k]
                # Find all i,j with s_sorted[i] + s_sorted[j] < threshold
                # Use two-pointer on sorted array
                # For each i, find the largest j such that s_sorted[i] + s_sorted[j] < threshold
                # Then sum w_sorted[i] * sum_{j' <= j} w_sorted[j']
                
                # Precompute prefix sums of w_sorted
                prefix_w = np.concatenate([[0.0], np.cumsum(w_sorted)])
                
                inner = 0.0
                j_ptr = n - 1
                for i in range(n):
                    # s_sorted[i] + s_sorted[j] < threshold
                    # s_sorted[j] < threshold - s_sorted[i]
                    target = threshold - s_sorted[i]
                    # Find rightmost index where s_sorted < target
                    # Since s_sorted is sorted, use binary search
                    j_max = np.searchsorted(s_sorted, target, side='left') - 1
                    if j_max >= 0:
                        inner += w_sorted[i] * prefix_w[j_max + 1]
                total += w[k] * w[l] * inner
        return total


def _compute_P_fast(support, weights):
    """Faster computation using precomputed pairwise sums."""
    n = len(support)
    if n == 0:
        return 0.0
    w = weights / weights.sum()
    s = support
    
    # Precompute all pairwise sums and their weights
    # sum_weight[i,j] = w[i]*w[j] for s[i]+s[j]
    # We need: for each (k,l), sum over (i,j) with s[i]+s[j] < 2*s[l]-s[k] of w[i]*w[j]
    
    # Create array of all pairwise sums
    s1, s2 = np.meshgrid(s, s, indexing='ij')
    pair_sums = s1 + s2  # shape (n, n)
    pair_weights = np.outer(w, w)  # shape (n, n)
    
    total = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            mask = pair_sums < threshold
            inner = np.sum(pair_weights * mask)
            total += w[k] * w[l] * inner
    return total


def _compute_P_fastest(support, weights):
    """Even faster using vectorized operations."""
    n = len(support)
    if n == 0:
        return 0.0
    w = weights / weights.sum()
    s = support
    
    # Precompute pairwise sums
    s1, s2 = np.meshgrid(s, s, indexing='ij')
    pair_sums = s1 + s2  # shape (n, n)
    pair_weights = np.outer(w, w)  # shape (n, n)
    
    # For each (k,l), threshold = 2*s[l] - s[k]
    # We can vectorize over k and l
    # thresholds shape: (n, n) where thresholds[k,l] = 2*s[l] - s[k]
    # s_k = s[:, None]  # (n, 1)
    # s_l = s[None, :]  # (1, n)
    # thresholds = 2*s_l - s_k  # (n, n)
    
    s_k = s[:, None]  # (n, 1)
    s_l = s[None, :]  # (1, n)
    thresholds = 2.0 * s_l - s_k  # (n, n)
    
    # For each (k,l), need sum of pair_weights where pair_sums < thresholds[k,l]
    # This is a bit tricky to fully vectorize without O(n^4)
    
    # Alternative: sort pair_sums and use cumulative sums
    flat_sums = pair_sums.flatten()
    flat_weights = pair_weights.flatten()
    
    # Sort by pair_sums
    order = np.argsort(flat_sums)
    sorted_sums = flat_sums[order]
    sorted_weights = flat_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    total = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Find number of elements < threshold
            idx = np.searchsorted(sorted_sums, threshold, side='left')
            inner = cum_weights[idx - 1] if idx > 0 else 0.0
            total += w[k] * w[l] * inner
    return total


def _optimize_3point():
    """Optimize a 3-point distribution: support [a, b, c] with weights [p, q, 1-p-q]"""
    from scipy.optimize import minimize
    
    def neg_P(params):
        a, b, c, p, q = params
        # Ensure validity
        if a < 0 or b < 0 or c < 0:
            return 1e10
        if p < 0 or q < 0 or p + q > 1:
            return 1e10
        if a > b or b > c:
            return 1e10
        
        support = np.array([a, b, c])
        weights = np.array([p, q, 1.0 - p - q])
        
        P = _compute_P_fast(support, weights)
        return -P
    
    # Try multiple initial guesses
    best_P = -1e10
    best_params = None
    
    initial_guesses = [
        [0.0, 0.5, 1.0, 0.5, 0.3],
        [0.0, 0.3, 1.0, 0.6, 0.2],
        [0.0, 1.0, 2.0, 0.5, 0.3],
        [0.0, 0.5, 2.0, 0.4, 0.4],
        [0.0, 0.2, 1.0, 0.7, 0.1],
        [0.0, 0.8, 1.0, 0.5, 0.3],
        [0.0, 0.1, 0.5, 0.6, 0.3],
        [0.0, 0.5, 1.5, 0.5, 0.25],
    ]
    
    for guess in initial_guesses:
        try:
            result = minimize(neg_P, guess, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            P_val = -result.fun
            if P_val > best_P:
                best_P = P_val
                best_params = result.x
        except Exception:
            pass
    
    if best_params is not None:
        a, b, c, p, q = best_params
        a = max(a, 0.0)
        b = max(b, a)
        c = max(c, b)
        p = max(0.0, min(p, 1.0))
        q = max(0.0, min(q, 1.0 - p))
        support = np.array([a, b, c])
        weights = np.array([p, q, 1.0 - p - q])
        return support, weights, best_P
    else:
        # Fallback
        support = np.array([0.0, 0.5, 1.0])
        weights = np.array([0.5, 0.3, 0.2])
        P = _compute_P_fast(support, weights)
        return support, weights, P


def _optimize_2point():
    """Optimize a 2-point distribution."""
    from scipy.optimize import minimize
    
    def neg_P(params):
        a, b, p = params
        if a < 0 or b < 0:
            return 1e10
        if a > b:
            a, b = b, a
        if p < 0 or p > 1:
            return 1e10
        support = np.array([a, b])
        weights = np.array([p, 1.0 - p])
        P = _compute_P_fast(support, weights)
        return -P
    
    best_P = -1e10
    best_params = None
    
    for guess in [[0.0, 1.0, 0.5], [0.0, 2.0, 0.6], [0.0, 0.5, 0.7], [0.0, 1.0, 0.8]]:
        try:
            result = minimize(neg_P, guess, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            P_val = -result.fun
            if P_val > best_P:
                best_P = P_val
                best_params = result.x
        except Exception:
            pass
    
    if best_params is not None:
        a, b, p = best_params
        a = max(a, 0.0)
        b = max(b, 0.0)
        if a > b:
            a, b = b, a
        p = max(0.0, min(p, 1.0))
        support = np.array([a, b])
        weights = np.array([p, 1.0 - p])
        return support, weights, best_P
    else:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        P = _compute_P_fast(support, weights)
        return support, weights, P


def _optimize_4point():
    """Optimize a 4-point distribution."""
    from scipy.optimize import minimize
    
    def neg_P(params):
        a, b, c, d, p, q, r = params
        s = np.array([a, b, c, d])
        if np.any(s < 0):
            return 1e10
        w = np.array([p, q, r, 1.0 - p - q - r])
        if np.any(w < 0):
            return 1e10
        if w.sum() <= 1e-10:
            return 1e10
        P = _compute_P_fast(s, w)
        return -P
    
    best_P = -1e10
    best_params = None
    
    guesses = [
        [0.0, 0.3, 0.6, 1.0, 0.4, 0.3, 0.2],
        [0.0, 0.5, 1.0, 1.5, 0.4, 0.3, 0.2],
        [0.0, 0.2, 0.8, 1.0, 0.5, 0.25, 0.15],
    ]
    
    for guess in guesses:
        try:
            result = minimize(neg_P, guess, method='Nelder-Mead',
                             options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10})
            P_val = -result.fun
            if P_val > best_P:
                best_P = P_val
                best_params = result.x
        except Exception:
            pass
    
    if best_params is not None:
        a, b, c, d, p, q, r = best_params
        s = np.array([a, b, c, d])
        w = np.array([p, q, r, 1.0 - p - q - r])
        s = np.maximum(s, 0.0)
        w = np.maximum(w, 0.0)
        if w.sum() <= 1e-10:
            w = np.ones(4) / 4.0
        return s, w, best_P
    else:
        support = np.array([0.0, 0.3, 0.6, 1.0])
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        P = _compute_P_fast(support, weights)
        return support, weights, P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized;
            the evaluator normalizes them to sum to 1).
    """
    start_time = time.time()
    deadline = start_time + 600
    
    # Start with 3-point optimization (known to be good from knowledge)
    support3, weights3, P3 = _optimize_3point()
    best_support = support3.copy()
    best_weights = weights3.copy()
    best_P = P3
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 100:
        # Try 2-point
        support2, weights2, P2 = _optimize_2point()
        if P2 > best_P:
            best_support = support2.copy()
            best_weights = weights2.copy()
            best_P = P2
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 80:
        # Try 4-point
        support4, weights4, P4 = _optimize_4point()
        if P4 > best_P:
            best_support = support4.copy()
            best_weights = weights4.copy()
            best_P = P4
    
    # Try some grid-based optimizations
    elapsed = time.time() - start_time
    if elapsed < deadline - 50:
        # Try uniform on [0, n-1] for various n
        for n in [2, 3, 4, 5, 6, 8, 10, 12]:
            if time.time() > deadline - 30:
                break
            s = np.arange(n, dtype=np.float64)
            w = np.ones(n, dtype=np.float64) / n
            P_val = _compute_P_fast(s, w)
            if P_val > best_P:
                best_support = s.copy()
                best_weights = w.copy()
                best_P = P_val
    
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Check for NaN/inf
    if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)):
        best_support = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        best_weights = np.array([0.4, 0.35, 0.25], dtype=np.float64)
    if np.any(np.isnan(best_weights)) or np.any(np.isinf(best_weights)):
        best_weights = np.ones_like(best_support) / len(best_support)
    
    # Ensure non-negative support
    best_support = np.maximum(best_support, 0.0)
    
    # Ensure non-negative weights
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure sum of weights > 0
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_support) / len(best_support)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
