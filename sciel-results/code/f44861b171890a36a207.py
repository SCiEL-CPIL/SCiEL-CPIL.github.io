# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 590  # 590s deadline, 10s safety margin


def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] exactly in float64.
    Uses vectorized computation with sorting for efficiency.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Pair sums and pair probabilities
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row  # shape (n, n)
    pair_probs = np.outer(w, w)  # shape (n, n)
    
    # For each (k, l) pair, threshold = 2*s_l - s_k
    # We need: sum_{k,l} w_k * w_l * (sum_{i,j} pair_probs[i,j] * I(pair_sums[i,j] < threshold))
    
    # Vectorize: create arrays of shape (n, n, n) for k, l and pair (i,j)
    # But that's n^3 which could be large. Let's use a smarter approach.
    
    # For each l, we need sum_k w_k * I(s_i + s_j < 2*s_l - s_k)
    # = sum_k w_k * I(s_k < 2*s_l - s_i - s_j)
    
    # So P = sum_{i,j,l} w_i * w_j * w_l * (sum_k w_k * I(s_k < 2*s_l - s_i - s_j))
    
    # Let's sort support and use cumulative sums for the inner sum
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    w_cumsum = np.cumsum(w_sorted)
    
    # For each (i, j, l), threshold_k = 2*s_l - s_i - s_j
    # inner_sum = sum of w_k for s_k < threshold_k = w_cumsum[searchsorted(s_sorted, threshold_k, side='left') - 1]
    
    # This is still n^3 operations. For n up to ~50, that's 125K which is fine.
    # Let's vectorize over (i, j, l)
    
    # Build (i, j, l) indices
    ii, jj, ll = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    ii = ii.ravel()
    jj = jj.ravel()
    ll = ll.ravel()
    
    thresholds = 2.0 * s[ll] - s[ii] - s[jj]  # shape (n^3,)
    
    # For each threshold, find how many sorted support values are < threshold
    idx = np.searchsorted(s_sorted, thresholds, side='left')
    # inner_sum[k] = sum of w_sorted[0:idx[k]]
    # Use cumsum with padding
    w_cumsum_padded = np.concatenate([[0.0], w_cumsum])
    inner_sums = w_cumsum_padded[idx]
    
    # P = sum over all (i,j,l) of w_i * w_j * w_l * inner_sums
    P = np.sum(w[ii] * w[jj] * w[ll] * inner_sums)
    
    return float(P)


def _compute_P_fast(support, weights):
    """
    Alternative faster computation using 2D approach.
    P = sum_{k,l} w_k * w_l * C(2*s_l - s_k)
    where C(t) = sum_{i,j} w_i * w_j * I(s_i + s_j < t)
    
    Precompute pair_sums and pair_probs, then for each (k,l) do a sum.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Pair sums and probabilities
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row  # (n, n)
    pair_probs = np.outer(w, w)  # (n, n)
    
    # Sort pair_sums for efficient CDF lookup
    ps_flat = pair_sums.ravel()
    pp_flat = pair_probs.ravel()
    order = np.argsort(ps_flat)
    ps_sorted = ps_flat[order]
    pp_sorted = pp_flat[order]
    pp_cumsum = np.concatenate([[0.0], np.cumsum(pp_sorted)])
    
    # For each (k, l): threshold = 2*s_l - s_k
    # C(threshold) = pp_cumsum[searchsorted(ps_sorted, threshold, side='left')]
    
    # Vectorize over (k, l)
    kk, ll = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    kk = kk.ravel()
    ll = ll.ravel()
    
    thresholds = 2.0 * s[ll] - s[kk]  # shape (n^2,)
    idx = np.searchsorted(ps_sorted, thresholds, side='left')
    C_vals = pp_cumsum[idx]  # shape (n^2,)
    
    # P = sum_{k,l} w_k * w_l * C_vals
    P = np.sum(w[kk] * w[ll] * C_vals)
    
    return float(P)


def _neg_P(params, support_fixed=None, weights_fixed=None):
    """Objective function for optimization. params contains both support and weights."""
    n = len(params) // 2
    s = params[:n]
    w = params[n:]
    
    # Ensure non-negative support and weights
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 1e-15)
    
    return -_compute_P_fast(s, w)


def _optimize_from_init(support0, weights0, max_iter=2000, timeout=30):
    """Run Nelder-Mead optimization from initial point."""
    n = len(support0)
    x0 = np.concatenate([support0, weights0])
    
    start_time = time.time()
    
    try:
        result = minimize(
            _neg_P, x0, method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
    except Exception:
        return support0, weights0, _compute_P_fast(support0, weights0)
    
    elapsed = time.time() - start_time
    if elapsed > timeout:
        return support0, weights0, _compute_P_fast(support0, weights0)
    
    x_opt = result.x
    s_opt = np.maximum(x_opt[:n], 0.0)
    w_opt = np.maximum(x_opt[n:], 1e-15)
    
    P_opt = _compute_P_fast(s_opt, w_opt)
    P_init = _compute_P_fast(support0, weights0)
    
    if P_opt > P_init:
        return s_opt, w_opt, P_opt
    else:
        return support0, weights0, P_init


def _try_two_point(p):
    """Try two-point distribution {0, L} with weight p at 0."""
    # We need to optimize over L and p
    # P = (1-p)*p^2*(3-2p) as derived, independent of L (as long as L > 0)
    # Actually wait, that's only if L is large enough. Let me re-derive.
    # For {0, L}: when X4=L, need X1+X2+X3 < 2L
    # If all at 0: 0 < 2L ✓
    # If one at L: L < 2L ✓  
    # If two at L: 2L < 2L ✗
    # If three at L: 3L < 2L ✗
    # So P = (1-p)[p^3 + 3p^2(1-p)] = (1-p)p^2(3-2p)
    # This is independent of L! So any L > 0 works.
    # Optimal p = (15-sqrt(33))/16 ≈ 0.5784
    # P ≈ 0.2589
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0
    return np.array([0.0, 1.0]), np.array([p_opt, 1.0 - p_opt])


def _try_three_point():
    """Try three-point distribution {0, a, b}."""
    # This is more complex. Let's do a grid search.
    best_s = None
    best_w = None
    best_P = -1.0
    
    for a in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for b in [1.5, 2.0, 3.0, 5.0, 8.0, 10.0]:
            if b <= a:
                continue
            for p0 in np.linspace(0.1, 0.9, 19):
                for p1 in np.linspace(0.05, 0.9 - p0, 15):
                    if p1 <= 0:
                        continue
                    p2 = 1.0 - p0 - p1
                    if p2 <= 0:
                        continue
                    s = np.array([0.0, a, b])
                    w = np.array([p0, p1, p2])
                    P = _compute_P_fast(s, w)
                    if P > best_P:
                        best_P = P
                        best_s = s.copy()
                        best_w = w.copy()
    
    return best_s, best_w


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    global _DEADLINE
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    # Baseline: uniform on {0, ..., 31}
    n_base = 32
    s_base = np.arange(n_base, dtype=np.float64)
    w_base = np.ones(n_base, dtype=np.float64) / n_base
    P_base = _compute_P_fast(s_base, w_base)
    best_support = s_base.copy()
    best_weights = w_base.copy()
    best_P = P_base
    
    # Try two-point
    if time.time() < _DEADLINE:
        s2, w2 = _try_two_point(0.5)
        P2 = _compute_P_fast(s2, w2)
        if P2 > best_P:
            best_P = P2
            best_support = s2.copy()
            best_weights = w2.copy()
    
    # Try three-point grid
    if time.time() < _DEADLINE:
        s3, w3 = _try_three_point()
        if s3 is not None:
            P3 = _compute_P_fast(s3, w3)
            if P3 > best_P:
                best_P = P3
                best_support = s3.copy()
                best_weights = w3.copy()
    
    # Try geometric-like distributions
    if time.time() < _DEADLINE:
        for n in [4, 5, 6, 8, 10]:
            for ratio in [1.5, 2.0, 3.0, 4.0]:
                s = np.array([0.0] + [ratio**i for i in range(1, n)])
                # Geometric weights
                r = 0.7
                w = np.array([r**i for i in range(n)])
                w = w / w.sum()
                P = _compute_P_fast(s, w)
                if P > best_P:
                    best_P = P
                    best_support = s.copy()
                    best_weights = w.copy()
    
    # Random search over small support sets
    if time.time() < _DEADLINE:
        np.random.seed(42)
        for trial in range(200):
            if time.time() > _DEADLINE - 10:
                break
            n = np.random.choice([2, 3, 4, 5, 6])
            # Random support: sorted, non-negative
            s = np.sort(np.random.exponential(1.0, n))
            s = s / s[-1] * np.random.uniform(1.0, 10.0)  # scale
            # Random Dirichlet weights
            w = np.random.dirichlet(np.ones(n) * np.random.uniform(0.5, 3.0))
            P = _compute_P_fast(s, w)
            if P > best_P:
                best_P = P
                best_support = s.copy()
                best_weights = w.copy()
    
    # Nelder-Mead refinement on best found
    if time.time() < _DEADLINE - 5:
        n = len(best_support)
        s_opt, w_opt, P_opt = _optimize_from_init(
            best_support, best_weights, max_iter=3000, timeout=45
        )
        if P_opt > best_P:
            best_P = P_opt
            best_support = s_opt.copy()
            best_weights = w_opt.copy()
    
    # Final verification
    P_final = _compute_P_fast(best_support, best_weights)
    
    # Ensure validity
    assert len(best_support) == len(best_weights)
    assert len(best_support) >= 1
    assert not np.any(np.isnan(best_support))
    assert not np.any(np.isinf(best_support))
    assert not np.any(np.isnan(best_weights))
    assert not np.any(np.isinf(best_weights))
    assert np.all(best_support >= -1e-9)
    assert np.all(best_weights >= -1e-9)
    assert np.sum(best_weights) > 1e-9
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END
