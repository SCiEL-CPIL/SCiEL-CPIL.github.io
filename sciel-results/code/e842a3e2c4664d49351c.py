import numpy as np
import time
import sys

def _compute_score_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
    Uses sorting and searchsorted for O(n^2 log n) evaluation.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w = weights.astype(np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    w = w / total
    s = support.astype(np.float64)
    
    # Pairwise sums s_i + s_j and their probabilities w_i * w_j
    ss = s[:, None] + s[None, :]
    ww = w[:, None] * w[None, :]
    
    pair_sums = ss.flatten()
    pair_weights = ww.flatten()
    
    # Sort pair sums
    order = np.argsort(pair_sums, kind='stable')
    sorted_sums = pair_sums[order]
    sorted_weights = pair_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    # Thresholds: for each (k, l), threshold is 2*s_l - s_k
    # We need F(threshold) = sum_{i,j: s_i+s_j < threshold} w_i w_j
    # Matrix of thresholds: T[k, l] = 2*s_l - s_k
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # searchsorted to find index of first element >= threshold (strictly less than threshold)
    idxs = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get cumulative sum up to idx-1 (or 0 if idx=0)
    # We want sum of weights where pair_sum < threshold
    # idxs is the index of the first element >= threshold
    # So we want cum_weights[idxs - 1]
    
    # Safe indexing
    idxs_clipped = np.clip(idxs - 1, 0, len(cum_weights) - 1)
    probs = np.where(idxs > 0, cum_weights[idxs_clipped], 0.0)
    
    # P = sum_{k,l} w_k w_l probs[k,l]
    P = np.sum(w[:, None] * w[None, :] * probs)
    
    return float(P)

def _compute_score_grad(support, weights):
    """
    Compute P and gradient dP/dw for fixed support.
    """
    n = len(support)
    if n == 0:
        return 0.0, np.array([])
    w = weights.astype(np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0, np.zeros(n)
    w = w / total
    s = support.astype(np.float64)
    
    # Precompute sorted pair sums and cum weights
    ss = s[:, None] + s[None, :]
    ww = w[:, None] * w[None, :]
    
    pair_sums = ss.flatten()
    pair_weights = ww.flatten()
    
    order = np.argsort(pair_sums, kind='stable')
    sorted_sums = pair_sums[order]
    sorted_weights = pair_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    # Thresholds
    thresholds = 2.0 * s[None, :] - s[:, None]
    idxs = np.searchsorted(sorted_sums, thresholds, side='left')
    idxs_clipped = np.clip(idxs - 1, 0, len(cum_weights) - 1)
    probs = np.where(idxs > 0, cum_weights[idxs_clipped], 0.0)
    
    # P = sum_{k,l} w_k w_l probs[k,l]
    P = np.sum(w[:, None] * w[None, :] * probs)
    
    # Gradient dP/dw_m
    # P = sum_{k,l,i,j} w_k w_l w_i w_j I(s_i+s_j < 2s_l - s_k)
    # dP/dw_m = sum_{k,l,i,j} [I(k=m) + I(l=m) + I(i=m) + I(j=m)] w_k w_l w_i w_j I(...)
    # Let Q_{kl} = sum_{i,j} w_i w_j I(s_i+s_j < 2s_l - s_k) = probs[k,l]
    # Term 1 (k=m): sum_{l} w_m w_l Q_{ml}
    # Term 2 (l=m): sum_{k} w_k w_m Q_{km}
    # Term 3 (i=m or j=m):
    #   Sum over k,l where m is i or j in the inner sum.
    #   Let R_{kl} = sum_{i,j} (I(i=m) + I(j=m)) w_i w_j I(s_i+s_j < 2s_l - s_k)
    #   Note: if i=m, term is w_m w_j. If j=m, term is w_i w_m.
    #   So R_{kl} = w_m * sum_{j: s_m+s_j < 2s_l-s_k} w_j + w_m * sum_{i: s_i+s_m < 2s_l-s_k} w_i
    #   Since s_i+s_j is symmetric, sum_{i: s_i+s_m < T} w_i is the same as sum_{j: s_m+s_j < T} w_j.
    #   Let H_{m, l, k} = sum_{j: s_m+s_j < 2s_l - s_k} w_j
    #   Then Term 3 contribution for (k,l) is w_k w_l * 2 * w_m * H_{m, l, k}
    #   Total Term 3 = sum_{k,l} w_k w_l * 2 * w_m * H_{m, l, k}
    
    # This gradient computation is O(n^3) if done naively for each m.
    # However, n is small (<= 20-30). O(n^3) is fine.
    
    grad = np.zeros(n)
    
    # Terms 1 & 2
    # T1[m] = sum_l w_l Q_{ml}
    # T2[m] = sum_k w_k Q_{km}
    T1 = w @ probs # sum_l w_l probs[m,l]
    T2 = probs.T @ w # sum_k probs[k,m] w_k -> sum_k w_k probs[k,m]
    
    # Term 3
    # For each m, we need sum_{k,l} w_k w_l * 2 * w_m * H_{m, l, k}
    # H_{m, l, k} = sum_{j: s_m+s_j < 2s_l - s_k} w_j
    
    # Precompute H for all m, l, k? That's n^3 values.
    # H[m, l, k]
    # Let's compute H efficiently.
    # For fixed m, we have array of values v_j = s_m + s_j.
    # We need sum of w_j where v_j < threshold_{lk} = 2s_l - s_k.
    
    # We can vectorize over k and l for each m.
    # Or just loop m.
    for m in range(n):
        # v_j = s_m + s_j
        v_j = s[m] + s
        # Sort v_j? No, s is sorted, so v_j is sorted.
        # We need sum of w_j for v_j < T.
        # Since v_j is sorted, we can use searchsorted.
        # But w_j is not sorted.
        # We need prefix sums of w_j corresponding to sorted v_j.
        # Since s is sorted, v_j is sorted in index order.
        # So we just need prefix sums of w.
        w_prefix = np.cumsum(w)
        
        # Thresholds for this m: T_{lk} = 2s_l - s_k
        # Shape (n_l, n_k)
        # We want H_{lk} = sum_{j: s_m+s_j < T_{lk}} w_j
        
        # Searchsorted on v_j (which is s[m] + s)
        # We want index of first element >= T.
        # Then sum w[0:idx].
        
        # Vectorized searchsorted
        # v_j is 1D array of size n.
        # T is 2D array of size n x n.
        # np.searchsorted(v_j, T, side='left') returns 2D array of indices.
        
        idxs_H = np.searchsorted(v_j, 2.0 * s[None, :] - s[:, None], side='left')
        
        # Get prefix sums
        # If idx is 0, sum is 0. Else sum is w_prefix[idx-1].
        idxs_H_clipped = np.clip(idxs_H - 1, 0, n - 1)
        H_vals = np.where(idxs_H > 0, w_prefix[idxs_H_clipped], 0.0)
        
        # H_vals has shape (n_k, n_l) ?
        # v_j is s[m] + s.
        # T is 2s_l - s_k.
        # searchsorted(v_j, T) -> for each T[i,j], find pos in v_j.
        # T shape: (n_l, n_k) if T = 2s[None,:] - s[:,None]
        # Wait, in code above: thresholds = 2.0 * s[None, :] - s[:, None]
        # This is shape (n_rows, n_cols) where row index is k (from s[:, None]) and col is l (from s[None, :])?
        # s[:, None] is (n, 1). s[None, :] is (1, n).
        # 2*s[None, :] is (1, n).
        # Result is (n, n). Entry [i, j] is 2*s[j] - s[i].
        # So i corresponds to k (X3), j corresponds to l (X4).
        # So H_vals[i, j] = H_{k=i, l=j}.
        
        # We need sum_{k,l} w_k w_l * 2 * w_m * H_{k,l}
        # Term3_m = 2 * w_m * sum_{k,l} w_k w_l H_{k,l}
        
        # sum_{k,l} w_k w_l H_{k,l} = sum_k w_k (sum_l w_l H_{k,l})
        # Let's compute S_m = sum_{k,l} w_k w_l H_vals[k,l]
        
        # H_vals is (n, n).
        # W_k is w[:, None]. W_l is w[None, :].
        # S_m = np.sum(w[:, None] * w[None, :] * H_vals)
        
        S_m = np.sum(w[:, None] * w[None, :] * H_vals)
        
        grad[m] = T1[m] + T2[m] + 2.0 * w[m] * S_m
        
    return float(P), grad

def _project_simplex(v):
    """Project v onto probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)

def _optimize_weights_fixed_support(support, initial_weights, max_iters=500, lr=0.1):
    """
    Optimize weights for fixed support using gradient ascent on simplex.
    """
    n = len(support)
    w = initial_weights.copy()
    w = w / w.sum()
    
    best_w = w.copy()
    best_score = _compute_score_fast(support, w)
    
    for it in range(max_iters):
        score, grad = _compute_score_grad(support, w)
        
        if score > best_score:
            best_score = score
            best_w = w.copy()
            
        # Gradient step
        w_new = w + lr * grad
        w_new = _project_simplex(w_new)
        
        # Check improvement
        score_new = _compute_score_fast(support, w_new)
        
        if score_new > score + 1e-12:
            w = w_new
            # Adaptive learning rate
            # if score_new > best_score * 1.001:
            #     lr *= 1.05
            # else:
            #     lr *= 0.95
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
                
    return best_w, best_score

def _optimize_support_fixed_weights(support, weights, max_iters=100, step=0.05):
    """
    Local search to optimize support points for fixed weights.
    """
    n = len(support)
    best_sup = support.copy()
    best_w = weights.copy()
    best_score = _compute_score_fast(best_sup, best_w)
    
    current_step = step
    stall = 0
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            if i == 0:
                # s[0] is usually 0, keep it? Or allow movement?
                # Task says support >= 0.
                # If we move s[0], we must maintain order.
                # Let's try moving all points.
                pass
            
            for direction in [-1, 1]:
                new_sup = best_sup.copy()
                new_sup[i] += direction * current_step
                
                # Check constraints
                if new_sup[i] < 0:
                    continue
                if i > 0 and new_sup[i] <= new_sup[i-1]:
                    continue
                if i < n-1 and new_sup[i] >= new_sup[i+1]:
                    continue
                    
                score = _compute_score_fast(new_sup, best_w)
                if score > best_score + 1e-12:
                    best_score = score
                    best_sup = new_sup
                    improved = True
                    
        if improved:
            stall = 0
        else:
            stall += 1
            current_step *= 0.5
            if current_step < 1e-6:
                break
            if stall > 10:
                break
                
    return best_sup, best_score

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Strategy:
    # 1. Start with a small number of points (e.g., 4-6)
    # 2. Optimize weights for fixed support
    # 3. Optimize support for fixed weights
    # 4. Iterate
    # 5. Try different initial configurations (e.g., uniform, geometric, random)
    
    # Initial candidates
    candidates = []
    
    # Uniform on {0, 1, ..., k}
    for k in [3, 4, 5, 6, 8, 10]:
        support = np.arange(k+1, dtype=np.float64)
        weights = np.ones(k+1, dtype=np.float64) / (k+1)
        candidates.append((support, weights))
        
    # Geometric
    for r in [0.5, 0.7, 0.8]:
        for k in [5, 10]:
            support = np.arange(k+1, dtype=np.float64)
            weights = r ** support
            weights /= weights.sum()
            candidates.append((support, weights))
            
    # Sparse: {0, a, b, ...}
    # Try some structured sparse sets
    for a in [1.0, 2.0, 3.0]:
        for b in [2.0, 4.0, 6.0, 10.0]:
            if a >= b: continue
            support = np.array([0.0, a, b])
            weights = np.array([0.3, 0.3, 0.4])
            candidates.append((support, weights))
            
    # Evaluate all candidates to find best starting point
    for support, weights in candidates:
        if time.time() > deadline - 60.0:
            break
        score = _compute_score_fast(support, weights)
        if score > best_score:
            best_score = score
            best_support = support.copy()
            best_weights = weights.copy()
            
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        best_score = _compute_score_fast(best_support, best_weights)
        
    # Refinement loop
    # Alternate between optimizing weights and support
    
    max_outer_iters = 20
    
    for outer_it in range(max_outer_iters):
        if time.time() > deadline - 30.0:
            break
            
        # 1. Optimize weights for fixed support
        # Use multiple initial weights if possible?
        # Just use current best weights.
        w_opt, score_w = _optimize_weights_fixed_support(
            best_support, best_weights, 
            max_iters=200, 
            lr=0.1
        )
        
        if score_w > best_score:
            best_score = score_w
            best_weights = w_opt
        
        # 2. Optimize support for fixed weights
        # Local search on support
        s_opt, score_s = _optimize_support_fixed_weights(
            best_support, best_weights,
            max_iters=50,
            step=0.1
        )
        
        if score_s > best_score:
            best_score = score_s
            best_support = s_opt
            
        # 3. Try adding a point?
        # If we have time, try to see if adding a point improves things.
        # This is harder. Let's stick to fixed N for now to ensure robustness.
        # Or try a few different N values in separate threads? No, single thread.
        
        # Let's try a few different N values in sequence if time permits.
        # But first, let's see if we can improve current N.
        
        # If we stall, try a different N?
        # For now, just keep iterating.
        
        # Check for convergence
        # If best_score hasn't improved in last few iterations, break?
        # Simple: run all outer iterations if time permits.
        
    # Final Polish
    if time.time() < deadline - 10.0:
        # High precision polish on support
        s_pol, score_pol = _optimize_support_fixed_weights(
            best_support, best_weights,
            max_iters=100,
            step=0.01
        )
        if score_pol > best_score:
            best_score = score_pol
            best_support = s_pol
            
        # Optimize weights again
        w_pol, score_pol_w = _optimize_weights_fixed_support(
            best_support, best_weights,
            max_iters=200,
            lr=0.05
        )
        if score_pol_w > best_score:
            best_score = score_pol_w
            best_weights = w_pol
            
    # Ensure validity
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Normalize weights
    w_sum = np.sum(best_weights)
    if w_sum <= 1e-9:
        best_weights = np.ones(len(best_support)) / len(best_support)
    else:
        best_weights = best_weights / w_sum
        
    # Check for NaN/Inf
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        # Fallback
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    return best_support, best_weights

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Score:", _compute_score_fast(s, w))
