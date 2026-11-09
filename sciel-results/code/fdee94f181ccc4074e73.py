# EVOLVE-BLOCK-START
import numpy as np
import time


def _evaluate(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(support, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    
    # pair_sums[i,j] = s[i] + s[j], pair_probs[i,j] = p[i]*p[j]
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    # For each (k, l): threshold = 2*s[l] - s[k]
    # contribution = p[k] * p[l] * sum_{i,j: pair_sums[i,j] < threshold} pair_probs[i,j]
    
    # Vectorize: for each (k,l) pair, compute the sum
    # thresholds[k,l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]  # shape (n, n), thresholds[k,l]
    
    # For each (k,l), we need sum of pair_probs where pair_sums < thresholds[k,l]
    # This is O(n^4) if done naively. For n up to ~50, this is 6.25M operations, fine.
    
    # Use sorting approach: sort pair_sums, then for each threshold, binary search
    flat_sums = pair_sums.ravel()
    flat_probs = pair_probs.ravel()
    order = np.argsort(flat_sums)
    sorted_sums = flat_sums[order]
    sorted_probs = flat_probs[order]
    cumsum_probs = np.cumsum(sorted_probs)
    
    # For each threshold, find count of elements < threshold
    # np.searchsorted gives index of first element >= threshold
    # So count = searchsorted(sorted_sums, threshold, side='left')
    # Sum = cumsum_probs[count-1] if count > 0 else 0
    
    flat_thresholds = thresholds.ravel()  # shape (n*n,)
    counts = np.searchsorted(sorted_sums, flat_thresholds, side='left')
    # sum of probs for indices < count
    sums_arr = np.zeros_like(flat_thresholds)
    mask = counts > 0
    sums_arr[mask] = cumsum_probs[counts[mask] - 1]
    
    sums_2d = sums_arr.reshape(n, n)  # sums_2d[k, l]
    
    # Total P = sum_{k,l} p[k]*p[l] * sums_2d[k,l]
    weight_matrix = p[:, None] * p[None, :]
    P = np.sum(weight_matrix * sums_2d)
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin
    
    # Start with a two-point distribution
    # p at 0, (1-p) at M
    # Optimize p and M
    
    best_P = -1
    best_support = None
    best_weights = None
    
    # Try a range of two-point distributions
    for M in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        for p in np.linspace(0.01, 0.99, 100):
            if time.time() > deadline - 60:
                break
            support = np.array([0.0, M], dtype=np.float64)
            weights = np.array([p, 1-p], dtype=np.float64)
            P = _evaluate(support, weights)
            if P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = weights.copy()
    
    if best_support is None:
        # Fallback to uniform
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights
    
    # Try local optimization: add a third point
    # Start from best two-point, try adding a point
    support = best_support.copy()
    weights = best_weights.copy()
    n = len(support)
    
    # Try adding a third point at various locations
    for new_pos in np.linspace(0.0, 2.0 * support[-1] if support[-1] > 0 else 1.0, 50):
        if time.time() > deadline - 60:
            break
        # Insert new point
        new_support = np.insert(support, n, new_pos)
        # Try different weight allocations: give small weight to new point
        for eps in np.linspace(0.01, 0.3, 20):
            # Redistribute: scale existing weights by (1-eps), new point gets eps
            new_weights = np.append(weights * (1 - eps), eps)
            P = _evaluate(new_support, new_weights)
            if P > best_P:
                best_P = P
                best_support = new_support.copy()
                best_weights = new_weights.copy()
    
    # Local optimization: perturb support and weights
    support = best_support.copy()
    weights = best_weights.copy()
    n = len(support)
    
    # Gradient-free local search
    step_size = 0.1
    for iteration in range(500):
        if time.time() > deadline - 30:
            break
        
        improved = False
        for i in range(n):
            # Try perturbing support[i]
            for delta in [-step_size, step_size]:
                new_support = support.copy()
                new_support[i] = max(0.0, support[i] + delta)
                P = _evaluate(new_support, weights)
                if P > best_P:
                    best_P = P
                    best_support = new_support.copy()
                    best_weights = weights.copy()
                    improved = True
                    break
            
            # Try perturbing weights[i]
            for delta in [-step_size * 0.1, step_size * 0.1]:
                new_weights = weights.copy()
                new_weights[i] = max(1e-6, weights[i] + delta)
                # Normalize
                new_weights = new_weights / new_weights.sum()
                P = _evaluate(support, new_weights)
                if P > best_P:
                    best_P = P
                    best_support = support.copy()
                    best_weights = new_weights.copy()
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-4:
                break
    
    # Return best found
    support = best_support
    weights = best_weights
    
    # Ensure validity
    if len(support) != len(weights):
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n
    if np.any(support < -1e-9) or np.any(weights < -1e-9) or np.sum(weights) <= 1e-9:
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n
    if np.any(np.isnan(support)) or np.any(np.isinf(support)) or np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n
    
    return support, weights


# EVOLVE-BLOCK-END
