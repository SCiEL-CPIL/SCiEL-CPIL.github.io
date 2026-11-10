# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P_fast(support, weights):
    """
    Fast exact computation of P = Prob[X1+X2+X3 < 2*X4] for i.i.d. discrete mu.
    Uses the formula: P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2 s_l - s_k} w_i w_j)
    We compute the pairwise sum distribution once, then for each (k,l) pair,
    we need the CDF of the pair sum at threshold 2*s_l - s_k.
    
    More efficiently: 
    Let T = X1 + X2. We need P[T + X3 < 2*X4].
    Actually the given formula is:
    P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)
    
    Here X1=s_k, X4=s_l, and (X2, X3) = (s_i, s_j) with s_i + s_j < 2*s_l - s_k.
    
    So P = sum_k sum_l w_k w_l * F_T(2*s_l - s_k)
    where F_T(t) = Prob[X1+X2 < t] is the CDF of the sum of two i.i.d. copies.
    
    We can precompute all pairwise sums and their probabilities.
    The support of T has at most n^2 points. We can build a sorted array of all s_i + s_j 
    and the corresponding probability mass.
    """
    n = len(support)
    # Build all pairwise sums and their weights
    # pair_sums[i, j] = support[i] + support[j]
    # pair_weights[i, j] = weights[i] * weights[j]
    
    # To use searchsorted efficiently, we flatten and sort.
    pair_sums = (support[:, None] + support[None, :]).ravel()
    pair_weights = (weights[:, None] * weights[None, :]).ravel()
    
    # Sort by pair_sums
    order = np.argsort(pair_sums, kind='mergesort')
    sorted_sums = pair_sums[order]
    sorted_weights = pair_weights[order]
    
    # Cumulative sum of weights gives the CDF (for strict inequality, we need sum of weights where sum < threshold)
    # Since we want strict inequality s_i + s_j < threshold, we need the cumulative sum up to (but not including) the first element >= threshold.
    # np.searchsorted with side='left' gives the index of the first element >= threshold.
    # So the CDF for strict < is cumsum[searchsorted_left - 1] (with 0 if index is 0).
    
    cumsum_weights = np.cumsum(sorted_weights)
    
    # Now compute P = sum_k sum_l w_k w_l * F_T(2*s_l - s_k)
    # For each (k, l), threshold = 2*s_l - s_k
    # We need to sum over all k, l.
    
    # To speed up, we can vectorize:
    # thresholds[k, l] = 2*support[l] - support[k]
    thresholds = 2.0 * support[None, :] - support[:, None]  # shape (n, n): thresholds[k, l]
    
    # For each threshold, find the index in sorted_sums
    idxs = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    
    # F_T(t) = cumsum_weights[idx-1] if idx > 0 else 0
    # Handle idx=0 case
    F_vals = np.zeros_like(idxs, dtype=np.float64)
    mask = idxs > 0
    F_vals[mask] = cumsum_weights[idxs[mask] - 1]
    
    F_vals = F_vals.reshape(n, n)
    
    # P = sum_k sum_l w_k * w_l * F_vals[k, l]
    # = sum_k w_k * (sum_l w_l * F_vals[k, l])
    P = np.sum(weights[:, None] * (weights[None, :] * F_vals))
    
    return P


def _compute_P(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly.
    """
    return _compute_P_fast(support, weights)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a mixture of a point mass at 0 and a heavy-tailed component.
    The idea is to make X4 large with some probability so that 2*X4 exceeds X1+X2+X3.
    
    We'll optimize over a parametric family:
    - A point mass at 0 with weight w0
    - A geometric-like or Pareto-like tail for the rest
    
    Actually, let's use a simpler parametric family that we can optimize:
    Support: {0, a, 2a, 3a, ..., (m-1)a} with weights that decay.
    Or even simpler: two-point mass {0, M} with weights (1-p, p).
    
    For two points: support = [0, M], weights = [1-p, p]
    P = Prob[X1+X2+X3 < 2*X4]
    X4 = 0 with prob 1-p, X4 = M with prob p.
    If X4 = 0: 2*X4 = 0. X1+X2+X3 >= 0 always, and < 0 never (since support >= 0). 
        Actually, if all are 0, sum = 0 which is not < 0. So P = 0 when X4=0.
    If X4 = M: 2*X4 = 2M. Need X1+X2+X3 < 2M.
        X1, X2, X3 are each 0 or M.
        Sum is k*M where k = number of M's among X1,X2,X3.
        k can be 0,1,2,3.
        k=0: sum=0 < 2M, yes. Prob = (1-p)^3
        k=1: sum=M < 2M, yes. Prob = 3p(1-p)^2
        k=2: sum=2M < 2M, no (strict). 
        k=3: sum=3M < 2M, no.
        So conditional on X4=M, prob = (1-p)^3 + 3p(1-p)^2 = (1-p)^2(1-p+3p) = (1-p)^2(1+2p)
    
    Total P = p * (1-p)^2 * (1+2p)
    
    Let's maximize f(p) = p(1-p)^2(1+2p) for p in [0,1].
    f(p) = p(1 - 2p + p^2)(1+2p) = p(1 + 2p - 2p - 4p^2 + p^2 + 2p^3) = p(1 - 3p^2 + 2p^3) = p - 3p^3 + 2p^4
    f'(p) = 1 - 9p^2 + 8p^3
    Set to 0: 8p^3 - 9p^2 + 1 = 0
    p=1 is a root: 8-9+1=0. Factor: (p-1)(8p^2 - p - 1) = 0
    8p^2 - p - 1 = 0 => p = (1 ± sqrt(1+32))/16 = (1 ± sqrt(33))/16
    sqrt(33) ≈ 5.7446
    p = (1 + 5.7446)/16 ≈ 6.7446/16 ≈ 0.4215
    p = (1 - 5.7446)/16 < 0, discard.
    
    So optimal p ≈ 0.4215, M can be any positive value (by symmetry, the ratio doesn't matter for two-point).
    f(0.4215) ≈ 0.4215 * (0.5785)^2 * (1 + 2*0.4215) = 0.4215 * 0.3347 * 1.843 ≈ 0.259
    
    The uniform on {0,...,31} gives P ≈ 0.25 (need to verify). So two-point might be slightly better.
    
    Let's try to do better with 3 points: {0, a, b} with b > a > 0.
    Or use a continuous optimization.
    
    Given the time budget, let's implement a local optimization over a parametric family.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 seconds
    
    # Start with the two-point solution
    p_opt = (1 + np.sqrt(33.0)) / 16.0
    M = 1.0  # scale doesn't matter
    support = np.array([0.0, M], dtype=np.float64)
    weights = np.array([1.0 - p_opt, p_opt], dtype=np.float64)
    best_P = _compute_P(support, weights)
    best_support = support.copy()
    best_weights = weights.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 60.0:
        return best_support, best_weights
    
    # Now try to improve with a 3-point or 4-point support
    # Parametric family: support = [0, a, b] with weights [w0, wa, wb]
    # Or support = [0, a, 2a, 3a] with geometric weights.
    
    # Let's try a geometric family: support = [0, a, 2a, ..., (n-1)a] with weights proportional to r^k
    # Optimize over r in (0, 1) and n.
    
    best_P_found = best_P
    best_params = None
    
    for n in [3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 64]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60.0:
            break
        
        # Try geometric weights: w_k = (1-r) * r^k for k=0,...,n-1, then truncate and renormalize
        # Actually, let's just use w_k = r^k for k=0,...,n-1 and normalize.
        
        # Search over r in (0, 1)
        # Use a grid search first, then local refinement.
        
        # For small n, we can afford a finer grid.
        num_r_samples = 50 if n <= 10 else 30
        
        r_values = np.linspace(0.01, 0.99, num_r_samples)
        
        for r in r_values:
            elapsed = time.time() - start_time
            if elapsed > deadline - 60.0:
                break
            
            support = np.arange(n, dtype=np.float64)
            weights = r ** np.arange(n, dtype=np.float64)
            weights /= weights.sum()
            
            P = _compute_P(support, weights)
            if P > best_P_found:
                best_P_found = P
                best_params = (n, r)
                best_support = support.copy()
                best_weights = weights.copy()
    
    # Try a two-point family with different M (though scale shouldn't matter)
    # Already covered.
    
    # Try a 3-point family: {0, a, b} with optimized weights
    # This is more complex. Let's skip for now if time is tight.
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30.0:
        return best_support, best_weights
    
    # Final polish: if we found a good geometric family, try to optimize the support spacing.
    # For now, return the best found.
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
