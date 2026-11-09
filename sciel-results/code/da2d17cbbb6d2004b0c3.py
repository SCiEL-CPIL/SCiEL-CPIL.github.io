import numpy as np
import time
import itertools

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy:
    1. Analytical reduction to a 3-point support {0, 1, M}.
    2. Vectorized 2D grid search over M (log-spaced) and weight at 1 (uniform).
    3. Local refinement using Nelder-Mead on the best candidate.
    4. Fallback to a known good distribution if time runs out or errors occur.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    def calc_prob_vectorized(support, weights):
        """
        Compute P[X1 + X2 + X3 < 2*X4] efficiently using broadcasting.
        support: 1D array of shape (n,)
        weights: 1D array of shape (n,)
        """
        # Normalize weights to ensure they sum to 1 (safety)
        total_w = np.sum(weights)
        if total_w <= 1e-9:
            return 0.0
        w = weights / total_w
        
        n = len(support)
        if n == 0:
            return 0.0
            
        # Precompute S2 and Prob2
        S2 = support[:, None] + support[None, :]
        Prob2 = w[:, None] * w[None, :]
        
        p_total = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[k] - support[l]
                # Sum Prob2 where S2 < threshold
                mask = S2 < threshold
                prob_sum = np.sum(Prob2[mask])
                p_total += w[k] * w[l] * prob_sum
                
        return p_total

    def is_valid(support, weights):
        if len(support) != len(weights):
            return False
        if len(support) < 1:
            return False
        if np.any(np.isnan(support)) or np.any(np.isinf(support)):
            return False
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            return False
        if np.any(support < -1e-9):
            return False
        if np.any(weights < -1e-9):
            return False
        if np.sum(weights) <= 1e-9:
            return False
        return True

    best_score = -np.inf
    best_support = np.array([0.0, 1.0, 10.0])
    best_weights = np.array([0.5, 0.3, 0.2])
    
    # Phase 1: Analytical Grid Search on 3-point support {0, 1, M}
    # We assume the optimal support is of the form {0, 1, M} with M > 1.
    # Weights are (w0, w1, wM) with w0 + w1 + wM = 1.
    # Due to symmetry/scaling, we can fix one point at 0 and another at 1.
    # The third point M varies.
    # The weights w1 and wM vary, with w0 = 1 - w1 - wM.
    
    # Let's vectorize the grid search.
    # M values: log-spaced from 1.1 to 1e4
    M_vals = np.logspace(np.log10(1.1), 4.0, 500)
    w1_vals = np.linspace(0.01, 0.99, 200)
    wM_vals = np.linspace(0.01, 0.99, 200)
    
    # We need to ensure w0 > 0 => w1 + wM < 1.
    # We will iterate over M and w1, and compute best wM or just grid search all 3.
    # To keep it fast, we can precompute the probability for each M, w1, wM.
    # However, 500 * 200 * 200 = 20 million evaluations might be too slow in Python loops.
    # Let's reduce grid size or vectorize the inner loop.
    
    # Refined grid for better resolution
    M_vals = np.logspace(np.log10(1.1), 4.0, 300)
    w1_vals = np.linspace(0.01, 0.99, 150)
    wM_vals = np.linspace(0.01, 0.99, 150)
    
    # For each M, we have a support [0, 1, M].
    # We can compute the probability for all (w1, wM) pairs at once?
    # P = sum_{k,l} w_k w_l * (sum_{i,j} w_i w_j 1[s_i+s_j < 2s_k - s_l])
    # This is a polynomial in w's of degree 4.
    # Let's just loop over M and vectorize over w1, wM.
    
    for M in M_vals:
        if time.time() > deadline - 10:
            break
        support = np.array([0.0, 1.0, M])
        # Weights w = [w0, w1, wM]
        # w0 = 1 - w1 - wM
        # We need w0 > 0, w1 > 0, wM > 0.
        
        # Create meshgrid for w1 and wM
        W1_grid, WM_grid = np.meshgrid(w1_vals, wM_vals, indexing='ij')
        W0_grid = 1.0 - W1_grid - WM_grid
        
        # Mask invalid weights
        valid_mask = (W0_grid > 1e-9) & (W1_grid > 1e-9) & (WM_grid > 1e-9)
        
        if not np.any(valid_mask):
            continue
            
        w0 = W0_grid[valid_mask]
        w1 = W1_grid[valid_mask]
        wM = WM_grid[valid_mask]
        
        # We have arrays w0, w1, wM of shape (N_valid,)
        # Construct weights array of shape (N_valid, 3)
        W = np.stack([w0, w1, wM], axis=-1)
        
        # Now compute probability for each weight vector.
        # This is still a loop over N_valid if we use calc_prob_vectorized.
        # Let's try to vectorize the probability calculation for fixed support.
        
        # S2 = [[0, 1, M], [1, 2, 1+M], [M, 1+M, 2M]]
        # S2_flat = [0, 1, M, 1, 2, 1+M, M, 1+M, 2M]
        # W_flat[i, j] = w_i * w_j
        
        # P = sum_{k,l} W_k W_l * (sum_{i,j} W_i W_j 1[S2[i,j] < 2*S_k - S_l])
        
        # Let's precompute the masks for each (k, l) pair.
        # There are 9 (k, l) pairs.
        # For each (k, l), we have a mask over the 9 (i, j) pairs.
        # The inner sum is a quadratic form in W.
        # Let Q_{kl} be the 3x3 matrix where Q_{kl}[i, j] = 1 if S2[i, j] < 2*S_k - S_l else 0.
        # Then inner_sum_{kl} = W^T Q_{kl} W.
        # P = sum_{k,l} W_k W_l (W^T Q_{kl} W).
        
        S = support
        S2 = S[:, None] + S[None, :]
        
        Qs = []
        for k in range(3):
            for l in range(3):
                threshold = 2.0 * S[k] - S[l]
                mask = S2 < threshold
                Qs.append(mask.astype(float))
        
        # Qs is a list of 9 3x3 matrices.
        # We want to compute P for each W in W (shape N_valid, 3).
        # P = sum_{k,l} W_k W_l * (W @ Q_{kl} @ W)
        
        # Let's compute W @ Q_{kl} @ W for all W at once.
        # W is (N, 3). Q is (3, 3).
        # WQ = W @ Q.T => (N, 3)
        # WQW = sum(W * WQ, axis=1) => (N,)
        
        P_vals = np.zeros(len(W))
        
        for idx, Q in enumerate(Qs):
            k = idx // 3
            l = idx % 3
            # W_k and W_l are columns of W
            Wk = W[:, k]
            Wl = W[:, l]
            
            # Compute W @ Q @ W
            WQ = W @ Q.T
            WQW = np.sum(W * WQ, axis=1)
            
            P_vals += Wk * Wl * WQW
            
        # Find best
        best_idx = np.argmax(P_vals)
        best_P = P_vals[best_idx]
        
        if best_P > best_score:
            best_score = best_P
            best_support = support.copy()
            best_weights = W[best_idx].copy()
            
    # Phase 2: Local Refinement
    if best_score > -np.inf:
        n_best = len(best_support)
        
        def objective(params):
            s_raw = params[:n_best]
            w_raw = params[n_best:-1]
            w_last = 1.0 - np.sum(w_raw)
            
            s = s_raw
            w = np.append(w_raw, w_last)
            
            if np.any(s < -1e-9) or np.any(w < -1e-9) or np.sum(w) <= 1e-9:
                return -np.inf
            
            p = calc_prob_vectorized(s, w)
            return -p
        
        init_params = np.concatenate([best_support, best_weights[:-1]])
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < -best_score:
                s_res = res.x[:n_best]
                w_res = np.append(res.x[n_best:-1], 1.0 - np.sum(res.x[n_best:-1]))
                
                s_res = np.maximum(s_res, 0.0)
                w_res = np.maximum(w_res, 0.0)
                
                w_sum = np.sum(w_res)
                if w_sum > 1e-9:
                    w_res = w_res / w_sum
                else:
                    w_res = np.ones(n_best) / n_best
                
                p_res = calc_prob_vectorized(s_res, w_res)
                
                if p_res > best_score:
                    best_score = p_res
                    best_support = s_res
                    best_weights = w_res
                    
        except Exception:
            pass

    # Final Validity Check and Formatting
    if not is_valid(best_support, best_weights):
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.5, 0.3, 0.2])

    support_out = np.asarray(best_support, dtype=np.float64).flatten()
    weights_out = np.asarray(best_weights, dtype=np.float64).flatten()
    
    return support_out, weights_out

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(s)
    print(w)
