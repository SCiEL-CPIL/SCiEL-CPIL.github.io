# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
    s = support
    w = weights
    n = len(s)
    # Normalize weights
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    # Sort by support value for efficient computation
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    
    # Precompute pairwise sums: for each pair (i,j), s_i + s_j
    # We need: for each k,l: w_k * w_l * (sum of w_i*w_j where s_i+s_j < 2*s_l - s_k)
    
    # Build all pairwise sums and their probabilities
    # pairs[i,j] = s_i + s_j, prob[i,j] = w_i * w_j
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    pair_probs = w_sorted[:, None] * w_sorted[None, :]
    
    # For efficiency, flatten and sort pair sums
    flat_sums = pair_sums.ravel()
    flat_probs = pair_probs.ravel()
    
    # Sort by pair sum
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Prefix sum of probabilities for fast range queries
    prefix_probs = np.cumsum(sorted_probs)
    
    # For each (k, l), find count of pairs with sum < 2*s_l - s_k
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s_sorted[l] - s_sorted[k]
            # Find rightmost index where sorted_sums < threshold (strict)
            idx = np.searchsorted(sorted_sums, threshold, side='left')
            if idx > 0:
                prob_sum = prefix_probs[idx - 1]
            else:
                prob_sum = 0.0
            P += w_sorted[k] * w_sorted[l] * prob_sum
    
    return P

def _neg_P(params, support):
    """Negative P for minimization. params are log-weights for positivity."""
    w = np.exp(params)
    return -_compute_P(support, w)

def _optimize_weights(support, w0, maxiter=200):
    """Optimize weights given fixed support using L-BFGS-B on log-weights."""
    # Initial log-weights
    lw0 = np.log(np.maximum(w0, 1e-15))
    
    result = minimize(_neg_P, lw0, args=(support,), method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
    
    w_opt = np.exp(result.x)
    P_opt = -result.fun
    return w_opt, P_opt

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_P = 0.0
    best_support = None
    best_weights = None
    
    # Strategy 1: Two-point {0, 1} with optimized weights
    # P = (1-p)^2 where p is weight on 1
    # Maximized at p=0, giving P=1.0? No wait...
    # If all mass on 0: X1+X2+X3=0, 2*X4=0, 0<0 is false. P=0.
    # If all mass on 1: X1+X2+X3=3, 2*X4=2, 3<2 is false. P=0.
    # Let's recompute: support {0,1}, weights (a, 1-a)
    # X1+X2+X3 can be 0,1,2,3 with probs a^3, 3a^2(1-a), 3a(1-a)^2, (1-a)^3
    # 2*X4 can be 0 or 2 with probs a, 1-a
    # P = P[X1+X2+X3=0]*P[X4=1] + P[X1+X2+X3=1]*P[X4=1] + P[X1+X2+X3=2]*P[X4=1]*0 + ...
    # Actually: need X1+X2+X3 < 2*X4
    # If X4=0: 2*X4=0, need X1+X2+X3 < 0, impossible since non-negative. P=0.
    # If X4=1: 2*X4=2, need X1+X2+X3 < 2, i.e., X1+X2+X3 in {0,1}
    # P[X1+X2+X3=0] = a^3, P[X1+X2+X3=1] = 3a^2(1-a)
    # P = (1-a) * (a^3 + 3a^2(1-a)) = (1-a) * a^2 * (a + 3(1-a)) = (1-a)*a^2*(3-2a)
    # Let f(a) = (1-a)*a^2*(3-2a) for a in [0,1]
    # f(0)=0, f(1)=0. Maximize: f'(a) = ...
    # Let's just optimize numerically
    
    support_2pt = np.array([0.0, 1.0])
    w0_2pt = np.array([0.5, 0.5])
    w_opt, P_opt = _optimize_weights(support_2pt, w0_2pt, maxiter=500)
    if P_opt > best_P:
        best_P = P_opt
        best_support = support_2pt.copy()
        best_weights = w_opt.copy()
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Strategy 2: Three-point {0, 1, L} with small weight on L
    for L in [10.0, 50.0, 100.0, 500.0]:
        if time.time() > deadline:
            break
        support_3pt = np.array([0.0, 1.0, L])
        # Try several initial weight configurations
        for w0 in [np.array([0.4, 0.4, 0.2]),
                   np.array([0.5, 0.4, 0.1]),
                   np.array([0.3, 0.3, 0.4]),
                   np.array([0.6, 0.3, 0.1]),
                   np.array([0.7, 0.2, 0.1])]:
            if time.time() > deadline:
                break
            try:
                w_opt, P_opt = _optimize_weights(support_3pt, w0, maxiter=300)
                if P_opt > best_P:
                    best_P = P_opt
                    best_support = support_3pt.copy()
                    best_weights = w_opt.copy()
            except Exception:
                pass
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Strategy 3: Four-point {0, 1, 2, L}
    for L in [20.0, 50.0, 100.0]:
        if time.time() > deadline:
            break
        support_4pt = np.array([0.0, 1.0, 2.0, L])
        for w0 in [np.array([0.3, 0.3, 0.2, 0.2]),
                   np.array([0.4, 0.3, 0.2, 0.1]),
                   np.array([0.5, 0.2, 0.1, 0.2])]:
            if time.time() > deadline:
                break
            try:
                w_opt, P_opt = _optimize_weights(support_4pt, w0, maxiter=200)
                if P_opt > best_P:
                    best_P = P_opt
                    best_support = support_4pt.copy()
                    best_weights = w_opt.copy()
            except Exception:
                pass
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Strategy 4: Try a spread support {0, 1, 2, 3} with optimized weights
    support_4pt_spread = np.array([0.0, 1.0, 2.0, 3.0])
    for w0 in [np.array([0.4, 0.3, 0.2, 0.1]),
               np.array([0.3, 0.3, 0.2, 0.2]),
               np.array([0.5, 0.2, 0.15, 0.15])]:
        if time.time() > deadline:
            break
        try:
            w_opt, P_opt = _optimize_weights(support_4pt_spread, w0, maxiter=300)
            if P_opt > best_P:
                best_P = P_opt
                best_support = support_4pt_spread.copy()
                best_weights = w_opt.copy()
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Strategy 5: Two-point {0, L} with optimized weights
    # P = (1-p) * P[X1+X2+X3 < 2L | X1,X2,X3 in {0,L}]
    # X1+X2+X3 can be 0, L, 2L, 3L
    # Need < 2L: so 0 or L
    # P[X1+X2+X3=0] = (1-p)^3, P[X1+X2+X3=L] = 3(1-p)^2*p
    # P = p * ((1-p)^3 + 3(1-p)^2*p) = p*(1-p)^2*((1-p) + 3p) = p*(1-p)^2*(1+2p)
    # Maximize g(p) = p*(1-p)^2*(1+2p)
    for L in [5.0, 10.0, 50.0]:
        if time.time() > deadline:
            break
        support_2pt_L = np.array([0.0, L])
        for p0 in [0.1, 0.2, 0.3, 0.5]:
            if time.time() > deadline:
                break
            w0 = np.array([1-p0, p0])
            try:
                w_opt, P_opt = _optimize_weights(support_2pt_L, w0, maxiter=500)
                if P_opt > best_P:
                    best_P = P_opt
                    best_support = support_2pt_L.copy()
                    best_weights = w_opt.copy()
            except Exception:
                pass
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Strategy 6: Three-point {0, a, L} with a < L, optimize a as well
    # For a few values of a and L
    for a in [0.5, 1.0, 2.0]:
        for L in [20.0, 50.0, 100.0]:
            if time.time() > deadline:
                break
            support_3pt_aL = np.array([0.0, a, L])
            for w0 in [np.array([0.4, 0.3, 0.3]),
                       np.array([0.5, 0.2, 0.3]),
                       np.array([0.3, 0.4, 0.3])]:
                if time.time() > deadline:
                    break
                try:
                    w_opt, P_opt = _optimize_weights(support_3pt_aL, w0, maxiter=200)
                    if P_opt > best_P:
                        best_P = P_opt
                        best_support = support_3pt_aL.copy()
                        best_weights = w_opt.copy()
                except Exception:
                    pass
    
    # Final validation
    if best_support is None or best_weights is None:
        # Fallback to uniform on {0,...,31}
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Clip any negative values from optimization
    best_weights = np.maximum(best_weights, 0.0)
    
    # Normalize
    w_sum = best_weights.sum()
    if w_sum <= 1e-9:
        n = len(best_support)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        best_weights = best_weights / w_sum
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
