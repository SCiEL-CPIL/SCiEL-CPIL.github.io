# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support

    # Distribution of S2 = X1 + X2
    s_i = s[:, None]
    s_j = s[None, :]
    sums_2 = s_i + s_j
    prods_2 = w[:, None] * w[None, :]

    sums_2_flat = sums_2.flatten()
    prods_2_flat = prods_2.flatten()

    unique_sums_2, inverse_indices_2 = np.unique(sums_2_flat, return_inverse=True)
    probs_2 = np.bincount(inverse_indices_2, weights=prods_2_flat)

    # Sort and compute CDF
    sort_idx = np.argsort(unique_sums_2)
    sorted_sums_2 = unique_sums_2[sort_idx]
    sorted_probs_2 = probs_2[sort_idx]
    cdf_2 = np.cumsum(sorted_probs_2)

    # P = sum_{k, l} w_k w_l P(S2 < 2 s_l - s_k)
    # Vectorize over k and l
    # Thresholds T[k,l] = 2*s[l] - s[k]
    # T has shape (n, n) where T[k, l] = 2*s[l] - s[k]
    T = 2.0 * s[None, :] - s[:, None]  # shape (n, n), T[k, l]

    # For each (k, l), find number of sorted_sums_2 < T[k, l]
    # np.searchsorted returns index of first element >= value
    # side='left' gives count of elements < T
    idx_arr = np.searchsorted(sorted_sums_2, T, side='left')
    # p_cond[k, l] = cdf_2[idx_arr[k,l] - 1] if idx_arr[k,l] > 0 else 0
    # Use np.where to handle idx=0 case
    p_cond = np.where(idx_arr > 0, cdf_2[np.maximum(idx_arr - 1, 0)], 0.0)

    # P = sum_{k,l} w[k] * w[l] * p_cond[k, l]
    P_total = np.sum(w[:, None] * w[None, :] * p_cond)

    return float(P_total)


def _neg_P_from_params(params):
    """Objective function for optimization (negative P to minimize)."""
    n = len(params) // 2
    support = params[:n]
    weights = params[n:]

    # Ensure non-negative support
    support = np.maximum(support, 0.0)
    # Ensure non-negative weights
    weights = np.maximum(weights, 0.0)

    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0

    try:
        P = _compute_P(support, weights)
    except Exception:
        return 0.0

    if not np.isfinite(P):
        return 0.0

    return -P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Returns a discrete measure maximizing P[X1+X2+X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    P_best = -1.0
    support_best = np.array([0.0], dtype=np.float64)
    weights_best = np.array([1.0], dtype=np.float64)

    def update_best(sup, wts, P_val):
        nonlocal P_best, support_best, weights_best
        if P_val > P_best:
            P_best = P_val
            support_best = sup.copy()
            weights_best = wts.copy()

    # 1. Try uniform distributions on {0, 1, ..., n-1}
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if time.time() > deadline:
            break
        support = np.linspace(0.0, 1.0, n)
        weights = np.ones(n, dtype=np.float64)
        P = _compute_P(support, weights)
        update_best(support, weights, P)

    # 2. Try Beta-like distributions (discretized)
    for n in [32, 64, 128, 256]:
        if time.time() > deadline:
            break
        for alpha in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
            for beta in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
                support = np.linspace(0.0, 1.0, n)
                x = support
                x_safe = np.clip(x, 1e-12, 1.0 - 1e-12)
                weights = (x_safe ** (alpha - 1)) * ((1.0 - x_safe) ** (beta - 1))
                if not np.all(np.isfinite(weights)):
                    continue
                if np.sum(weights) <= 1e-9:
                    continue
                P = _compute_P(support, weights)
                update_best(support, weights, P)

    # 3. Try 2-point {0, 1} with optimized p
    for p in [0.3, 0.4, 0.42, 0.45, 0.5, 0.6]:
        support = np.array([0.0, 1.0])
        weights = np.array([1-p, p])
        P = _compute_P(support, weights)
        update_best(support, weights, P)

    # 4. Try 3-point {0, a, 1}
    for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for p0 in [0.2, 0.3, 0.4]:
            for p1 in [0.2, 0.3, 0.4]:
                p2 = 1.0 - p0 - p1
                if p2 < 0: continue
                support = np.array([0.0, a, 1.0])
                weights = np.array([p0, p1, p2])
                P = _compute_P(support, weights)
                update_best(support, weights, P)

    # 5. Local optimization from multiple starting points
    # We'll optimize both support and weights jointly.
    # Since the problem is non-convex, we use multiple starting points.
    
    # Starting points: various structured distributions
    starting_points = []
    
    # Uniform on [0,1] with n=16
    n = 16
    starting_points.append(np.concatenate([np.linspace(0.0, 1.0, n), np.ones(n)/n]))
    
    # Beta(1, 2) discretized
    n = 16
    x = np.linspace(1e-12, 1.0 - 1e-12, n)
    w_beta = (1.0 - x) / x * 0.5  # Beta(1,2) proportional
    w_beta = w_beta / np.sum(w_beta)
    starting_points.append(np.concatenate([x, w_beta]))
    
    # Beta(2, 1) discretized
    x = np.linspace(1e-12, 1.0 - 1e-12, n)
    w_beta = x / (1.0 - x) * 0.5  # Beta(2,1) proportional
    w_beta = w_beta / np.sum(w_beta)
    starting_points.append(np.concatenate([x, w_beta]))
    
    # Two peaks: mass at 0 and 1
    n = 8
    support_2peak = np.concatenate([np.linspace(0.0, 0.1, 4), np.linspace(0.9, 1.0, 4)])
    weights_2peak = np.concatenate([np.ones(4)*0.3, np.ones(4)*0.2])
    weights_2peak = weights_2peak / np.sum(weights_2peak)
    starting_points.append(np.concatenate([support_2peak, weights_2peak]))
    
    # Three peaks: 0, 0.5, 1
    n = 9
    support_3peak = np.concatenate([np.linspace(0.0, 0.1, 3), np.linspace(0.45, 0.55, 3), np.linspace(0.9, 1.0, 3)])
    weights_3peak = np.concatenate([np.ones(3)*0.4, np.ones(3)*0.2, np.ones(3)*0.4])
    weights_3peak = weights_3peak / np.sum(weights_3peak)
    starting_points.append(np.concatenate([support_3peak, weights_3peak]))
    
    # Random starting points
    np.random.seed(42)
    for _ in range(5):
        n = np.random.randint(4, 20)
        support_rand = np.sort(np.random.uniform(0.0, 1.0, n))
        weights_rand = np.random.dirichlet(np.ones(n))
        starting_points.append(np.concatenate([support_rand, weights_rand]))
    
    for start_params in starting_points:
        if time.time() > deadline:
            break
        
        n = len(start_params) // 2
        
        try:
            result = so.minimize(
                _neg_P_from_params,
                start_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-6, 'xatol': 1e-6}
            )
            
            if result.success or result.fun < 0:
                params_opt = result.x
                sup_opt = np.maximum(params_opt[:n], 0.0)
                wts_opt = np.maximum(params_opt[n:], 0.0)
                w_sum = np.sum(wts_opt)
                
                if w_sum > 1e-9:
                    wts_opt = wts_opt / w_sum
                    P_opt = _compute_P(sup_opt, wts_opt)
                    if np.isfinite(P_opt) and P_opt > P_best:
                        update_best(sup_opt, wts_opt, P_opt)
        except Exception:
            pass

    # 6. Try linear programming approach for fixed support grids
    # For a fixed support grid, the problem is linear in weights.
    # P = sum_{i,j,k,l} w_i w_j w_k w_l * I(s_i + s_j + s_k < 2*s_l)
    # This is NOT linear in weights (it's quartic). So LP won't work directly.
    # But we can try a heuristic: fix three of the four variables and optimize the fourth.
    # This is complex, so we'll skip it for now.

    # Final validation
    if len(support_best) == 0 or len(weights_best) == 0:
        support_best = np.array([0.0], dtype=np.float64)
        weights_best = np.array([1.0], dtype=np.float64)

    # Ensure valid output
    support_best = np.asarray(support_best, dtype=np.float64).flatten()
    weights_best = np.asarray(weights_best, dtype=np.float64).flatten()

    # Remove any NaN or inf
    mask = np.isfinite(support_best) & np.isfinite(weights_best)
    support_best = support_best[mask]
    weights_best = weights_best[mask]

    # Ensure non-negative
    support_best = np.maximum(support_best, 0.0)
    weights_best = np.maximum(weights_best, 0.0)

    # Ensure sum of weights > 0
    if np.sum(weights_best) <= 1e-9:
        support_best = np.array([0.0], dtype=np.float64)
        weights_best = np.array([1.0], dtype=np.float64)

    return support_best, weights_best


# EVOLVE-BLOCK-END
