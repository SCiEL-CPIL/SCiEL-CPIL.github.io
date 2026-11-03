import numpy as np
import time

N = 50

def _compute_ratio(h):
    """Compute the ratio of squared L2 norm to L1 norm times supremum norm of autoconvolution."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    # Autoconvolution: c[k] = sum_i h[i]*h[i+k] for lag k
    # For discrete heights on equal intervals, the autoconvolution at lag 0 is sum h[i]^2
    # The supremum norm is the max absolute value of the autoconvolution
    # Compute full autoconvolution
    n = len(h)
    # c[0] = sum h[i]^2
    # c[k] for k>0: sum_{i=0}^{n-1-k} h[i]*h[i+k]
    # c[-k] = c[k] by symmetry
    # The max will likely be at lag 0 (sum of squares) or nearby
    # Compute all lags efficiently
    c = np.zeros(2 * n - 1)
    for k in range(n):
        c[k] = np.dot(h[:n-k], h[k:])
        if k > 0:
            c[2*n-1-k] = c[k]
    sup_norm = np.max(np.abs(c))
    if sup_norm < 1e-12:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Best-response dynamics: iteratively update each h_i
    # For a fixed set of other heights, what's the optimal h_i?
    # The ratio is: (sum_j h_j^2) / (sum_j h_j * sup_k |c_k|)
    # where c_k = sum_j h_j h_{j+k}
    
    # Key insight: if h_i contributes disproportionately to the peak autoconvolution value,
    # setting it to zero might improve the ratio. Otherwise, there's an optimal local value.
    
    h = np.ones(N).copy()
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            h_old = h[i]
            h[i] = 0.0
            
            # Try a few candidate values for h_i
            # Optimal value considering the rest of the array
            # Let S = sum_{j≠i} h_j, Q = sum_{j≠i} h_j^2
            # Let P = max_k |c_k| where c_k computed with h_i = 0
            # If we set h_i = x, then:
            #   new_l2_sq = Q + x^2
            #   new_l1 = S + x
            #   new autoconvolution: c'_k = c_k + x*(h_{i+k} + h_{i-k} if in range) + x^2 (at lag 0)
            #   This is complex, so let's try a few values
            
            S = np.sum(h)  # h[i] is 0 now
            Q = np.sum(h * h)
            
            # Compute autoconvolution with h_i = 0
            n = len(h)
            c = np.zeros(2 * n - 1)
            for k in range(n):
                c[k] = np.dot(h[:n-k], h[k:])
                if k > 0:
                    c[2*n-1-k] = c[k]
            sup_base = np.max(np.abs(c))
            
            # Try candidate values
            candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
            # Also try the "optimal" value: balance between increasing numerator and denominator
            # The contribution of x to sup_norm depends on neighboring values
            # Simplified: try values around what balances x^2/(S+x) vs sup increase
            
            best_local = h_old
            best_local_score = _compute_ratio(h)  # with h_i = 0
            
            for x in candidates:
                h[i] = x
                s = _compute_ratio(h)
                if s > best_local_score + 1e-12:
                    best_local_score = s
                    best_local = x
            
            h[i] = best_local
            if abs(h[i] - h_old) > 1e-6:
                improved = True
        
        if not improved:
            break
        
        # Check if we improved the global best
        current_score = _compute_ratio(h)
        if current_score > best_score:
            best_score = current_score
            best_h = h.copy()
    
    # Final check
    final_score = _compute_ratio(best_h)
    if final_score > best_score:
        best_score = final_score
    
    return list(best_h)
