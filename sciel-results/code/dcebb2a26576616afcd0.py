# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    w = weights / weights.sum()
    # Pairwise sums: s_i + s_j with weight w_i * w_j
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
    # Build all n^2 pairwise sums
    S = support[:, None] + support[None, :]  # (n, n)
    W = w[:, None] * w[None, :]  # (n, n)
    # For each (k, l), we need sum of W[i,j] where S[i,j] < 2*s_l - s_k
    # threshold[k,l] = 2*s_l - s_k
    thresh = 2.0 * support[None, :] - support[:, None]  # (n, n): thresh[k,l] = 2*s_l - s_k
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            t = thresh[k, l]
            if t <= 0:
                continue
            # Sum W[i,j] where S[i,j] < t
            mask = S < t
            P += w[k] * w[l] * W[mask].sum()
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = 630.0 - 10.0  # safety margin
    
    # Start with the two-point optimal: P(X=0)=p, P(X=1)=1-p, p≈0.5784
    p_opt = (15 - np.sqrt(33)) / 16.0  # ≈ 0.5784
    support = np.array([0.0, 1.0], dtype=np.float64)
    weights = np.array([p_opt, 1.0 - p_opt], dtype=np.float64)
    
    best_support = support.copy()
    best_weights = weights.copy()
    best_P = _compute_P(best_support, best_weights)
    
    # Try to improve with a 3-point distribution using grid search
    # Points: 0, a, 1 (or 0, 1, M)
    # Let's try 0, x, 1 with weights p0, px, p1
    
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # 3-point search: 0, a, 1
        best_P_3 = best_P
        best_params_3 = (best_support, best_weights)
        
        # Grid search over a in (0,1) and weights
        for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for p0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                rem = 1.0 - p0
                for px_frac in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
                    px = rem * px_frac
                    p1 = rem - px
                    if p0 < 0 or px < 0 or p1 < 0:
                        continue
                    s = np.array([0.0, a, 1.0], dtype=np.float64)
                    w = np.array([p0, px, p1], dtype=np.float64)
                    P = _compute_P(s, w)
                    if P > best_P_3:
                        best_P_3 = P
                        best_params_3 = (s.copy(), w.copy())
            
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break
        
        if best_P_3 > best_P:
            best_P = best_P_3
            best_support, best_weights = best_params_3
    
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Try 4-point: 0, a, b, 1
        best_P_4 = best_P
        best_params_4 = (best_support, best_weights)
        
        for a in [0.2, 0.4, 0.6, 0.8]:
            for b in [a+0.1, a+0.2, a+0.3, a+0.4]:
                if b >= 1.0:
                    continue
                for p0 in [0.2, 0.3, 0.4, 0.5]:
                    rem = 1.0 - p0
                    for px_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                        px = rem * px_frac
                        rem2 = rem - px
                        for py_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                            py = rem2 * py_frac
                            pz = rem2 - py
                            if any(x < 0 for x in [p0, px, py, pz]):
                                continue
                            s = np.array([0.0, a, b, 1.0], dtype=np.float64)
                            w = np.array([p0, px, py, pz], dtype=np.float64)
                            P = _compute_P(s, w)
                            if P > best_P_4:
                                best_P_4 = P
                                best_params_4 = (s.copy(), w.copy())
                
                elapsed = time.time() - start_time
                if elapsed > deadline:
                    break
            if elapsed > deadline:
                break
        
        if best_P_4 > best_P:
            best_P = best_P_4
            best_support, best_weights = best_params_4
    
    # Validate
    if len(best_support) != len(best_weights):
        return np.array([0.0]), np.array([1.0])
    if len(best_support) < 1:
        return np.array([0.0]), np.array([1.0])
    if np.any(~np.isfinite(best_support)) or np.any(~np.isfinite(best_weights)):
        return np.array([0.0]), np.array([1.0])
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        return np.array([0.0]), np.array([1.0])
    if best_weights.sum() <= 1e-9:
        return np.array([0.0]), np.array([1.0])
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
