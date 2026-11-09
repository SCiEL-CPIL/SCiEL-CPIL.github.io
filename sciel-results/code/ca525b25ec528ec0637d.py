import numpy as np
import time

def calc_P(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] where X1..X4 are i.i.d. from discrete measure mu.
    
    Uses the formula:
    P = sum_{k,l} w_k * w_l * (sum_{i,j} w_i * w_j * 1(s_i + s_j < 2*s_l - s_k))
    
    Parameters:
    -----------
    support : 1D array of length n
    weights : 1D array of length n
    
    Returns:
    --------
    float : The probability P
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    # Precompute pairwise sums S_ij = s_i + s_j, shape (n, n)
    S = s[:, None] + s[None, :]
    
    # R_kl = 2*s_l - s_k, shape (n, n)
    R = 2 * s[None, :] - s[:, None]
    
    # We need: P = sum_{k,l} w_k * w_l * (sum_{i,j} w_i * w_j * 1(S_ij < R_kl))
    
    # For each (k, l), compute the inner sum: sum_{i,j} w_i * w_j * 1(S_ij < R_kl)
    # This can be done by: for each threshold t = R_kl, compute sum of w_i*w_j where S_ij < t
    
    # Efficient approach: sort S and use cumulative sums
    # But we need weighted sums, not just counts.
    
    # Alternative: For small n, do it directly with broadcasting
    # S_exp: (n, n, 1, 1), R_exp: (1, 1, n, n)
    S_exp = S[:, :, None, None]  # (n, n, 1, 1)
    R_exp = R[None, None, :, :]  # (1, 1, n, n)
    
    mask = (S_exp < R_exp)  # (n, n, n, n)
    
    # W_prod[i,j,k,l] = w_i * w_j * w_k * w_l
    W_prod = (w[:, None, None, None] * 
              w[None, :, None, None] * 
              w[None, None, :, None] * 
              w[None, None, None, :])
    
    P = np.sum(W_prod * mask)
    return float(P)


def calc_P_efficient(support, weights):
    """
    More efficient computation using sorting and cumulative sums.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    # Pairwise sums S_ij = s_i + s_j
    S = s[:, None] + s[None, :]
    
    # R_kl = 2*s_l - s_k
    R = 2 * s[None, :] - s[:, None]
    
    # For each (k, l), we need: sum_{i,j} w_i * w_j * 1(S_ij < R_kl)
    
    # Flatten S and create corresponding weight products
    S_flat = S.flatten()  # n*n elements
    W_flat = (w[:, None] * w[None, :]).flatten()  # w_i * w_j for each pair
    
    # Sort S_flat and W_flat together
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    
    # Cumulative sum of W_sorted
    W_cumsum = np.cumsum(W_sorted)
    
    # For each R_kl, find the number of S_ij < R_kl using searchsorted
    # Then the inner sum is W_cumsum[idx-1] if idx > 0, else 0
    
    R_flat = R.flatten()  # n*n elements
    # For each R_flat[k], idx = number of S_sorted < R_flat[k]
    idx = np.searchsorted(S_sorted, R_flat, side='left')
    
    # inner_sum[k] = W_cumsum[idx[k]-1] if idx[k] > 0 else 0
    inner_sum = np.zeros_like(R_flat)
    positive_idx = idx > 0
    inner_sum[positive_idx] = W_cumsum[idx[positive_idx] - 1]
    
    inner_sum = inner_sum.reshape(n, n)
    
    # Now P = sum_{k,l} w_k * w_l * inner_sum[k,l]
    W_kl = (w[:, None] * w[None, :]).reshape(n, n)
    
    P = np.sum(W_kl * inner_sum)
    return float(P)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill
    
    best_P = -np.inf
    best_support = None
    best_weights = None
    
    def try_solution(support, weights):
        nonlocal best_P, best_support, best_weights
        try:
            # Validate
            s = np.asarray(support, dtype=np.float64).flatten()
            w = np.asarray(weights, dtype=np.float64).flatten()
            
            if len(s) != len(w) or len(s) < 1:
                return
            if np.any(np.isnan(s)) or np.any(np.isnan(w)):
                return
            if np.any(np.isinf(s)) or np.any(np.isinf(w)):
                return
            if np.any(s < -1e-9):
                return
            if np.any(w < -1e-9):
                return
            if np.sum(w) <= 1e-9:
                return
            
            # Compute P
            P = calc_P_efficient(s, w)
            
            if P > best_P:
                best_P = P
                best_support = s.copy()
                best_weights = w.copy()
        except Exception:
            pass
    
    # 1. Seed: uniform on {0, 1, ..., 31}
    n_seed = 32
    support_seed = np.arange(n_seed, dtype=np.float64)
    weights_seed = np.ones(n_seed, dtype=np.float64) / n_seed
    try_solution(support_seed, weights_seed)
    
    # 2. Optimized 2-point binary distribution
    # For support [a, b] and weights [p, 1-p]:
    # X takes value a with prob p, b with prob 1-p
    # X1+X2+X3 < 2*X4
    # Let's analyze: 2*X4 can be 2a, a+b, or 2b
    # X1+X2+X3 can be 3a, 2a+b, a+2b, or 3b
    
    # Case 1: X4=a (prob p), need X1+X2+X3 < 2a
    #   Only possible if X1=X2=X3=a, prob p^3
    # Case 2: X4=b (prob 1-p), need X1+X2+X3 < 2b
    #   Possible: all a (p^3), two a one b (3p^2(1-p)), one a two b (3p(1-p)^2)
    #   3b is NOT < 2b, so exclude
    #   Total: p^3 + 3p^2(1-p) + 3p(1-p)^2 = p^3 + 3p^2 - 3p^3 + 3p - 6p^2 + 3p - 3p^2... 
    #   Let me redo: p^3 + 3p^2(1-p) + 3p(1-p)^2
    #   = p^3 + 3p^2 - 3p^3 + 3p(1 - 2p + p^2)
    #   = p^3 + 3p^2 - 3p^3 + 3p - 6p^2 + 3p^3
    #   = p^3 - 3p^3 + 3p^3 + 3p^2 - 6p^2 + 3p
    #   = p^3 - 3p^2 + 3p
    
    # So P = p * p^3 + (1-p) * (p^3 - 3p^2 + 3p)
    #      = p^4 + (1-p)(p^3 - 3p^2 + 3p)
    #      = p^4 + p^3 - 3p^2 + 3p - p^4 + 3p^3 - 3p^2 + 3p^2
    #      = p^3 - 3p^2 + 3p + 3p^3 - 3p^2 + 3p^2
    #      = 4p^3 - 3p^2 + 3p
    
    # Wait, let me recompute more carefully.
    # P = p * Prob[X1+X2+X3 < 2a | X4=a] + (1-p) * Prob[X1+X2+X3 < 2b | X4=b]
    # Prob[X1+X2+X3 < 2a]: only (a,a,a) works. Prob = p^3
    # Prob[X1+X2+X3 < 2b]: 
    #   (a,a,a): 3a < 2b if a < 2b/3. If b > 0 and a >= 0, this depends.
    #   For a=0, b=1: 0 < 2, yes. 
    #   (a,a,b): 2a+b < 2b iff a < b. Yes if a < b.
    #   (a,b,b): a+2b < 2b iff a < 0. No if a >= 0.
    #   (b,b,b): 3b < 2b iff b < 0. No.
    
    # So for 0 <= a < b:
    # Prob[X1+X2+X3 < 2b] = p^3 + 3p^2(1-p)
    # P = p * p^3 + (1-p) * (p^3 + 3p^2(1-p))
    #   = p^4 + (1-p)*p^3 + 3p^2(1-p)^2
    #   = p^4 + p^3 - p^4 + 3p^2(1 - 2p + p^2)
    #   = p^3 + 3p^2 - 6p^3 + 3p^4
    #   = 3p^4 - 5p^3 + 3p^2
    
    # Maximize f(p) = 3p^4 - 5p^3 + 3p^2 on [0,1]
    # f'(p) = 12p^3 - 15p^2 + 6p = 3p(4p^2 - 5p + 2)
    # 4p^2 - 5p + 2 = 0 -> discriminant = 25 - 32 = -7 < 0
    # So f'(p) = 3p(4p^2 - 5p + 2), and 4p^2 - 5p + 2 > 0 for all p
    # Thus f'(p) >= 0 for p >= 0, with f'(p) = 0 only at p = 0
    # So f is increasing on [0,1], max at p=1: f(1) = 3 - 5 + 3 = 1
    
    # Hmm, that gives P=1 at p=1 (all mass at a). That's degenerate.
    # Let me just try some specific 2-point distributions numerically.
    
    for a, b in [(0.0, 1.0), (0.0, 2.0), (0.0, 0.5), (1.0, 2.0), (0.0, 0.1)]:
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            try_solution(np.array([a, b]), np.array([p, 1-p]))
    
    # 3. 3-point distributions
    for support_3 in [
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.3, 1.0]),
        np.array([0.0, 0.5, 2.0]),
        np.array([0.0, 1.0, 3.0]),
    ]:
        # Grid search over simplex
        res = 20
        for w0 in np.linspace(0.0, 1.0, res):
            for w1 in np.linspace(0.0, 1.0 - w0, res):
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                w2 = max(w2, 0.0)
                try_solution(support_3, np.array([w0, w1, w2]))
    
    # 4. 4-point distributions with grid search
    if time.time() < deadline - 100:
        support_4 = np.array([0.0, 0.5, 1.0, 2.0])
        res = 12
        for w0 in np.linspace(0.0, 1.0, res):
            for w1 in np.linspace(0.0, 1.0 - w0, res):
                for w2 in np.linspace(0.0, 1.0 - w0 - w1, res):
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < -1e-9:
                        continue
                    w3 = max(w3, 0.0)
                    try_solution(support_4, np.array([w0, w1, w2, w3]))
                    if time.time() > deadline - 50:
                        break
                if time.time() > deadline - 50:
                    break
            if time.time() > deadline - 50:
                break
    
    # 5. Random search for small n
    if time.time() < deadline - 60:
        np.random.seed(42)
        for n in [2, 3, 4, 5]:
            for _ in range(500):
                if time.time() > deadline - 30:
                    break
                # Random support in [0, 2]
                support_rand = np.sort(np.random.uniform(0, 2, n))
                # Random Dirichlet weights
                weights_rand = np.random.dirichlet(np.ones(n))
                try_solution(support_rand, weights_rand)
    
    # 6. Local refinement using scipy
    from scipy.optimize import minimize
    
    if best_support is not None and time.time() < deadline - 30:
        n_best = len(best_support)
        
        def neg_P(params):
            try:
                if len(params) != 2 * n_best:
                    return 1.0
                s = params[:n_best]
                w = params[n_best:]
                
                # Ensure s is sorted and non-negative
                s = np.maximum(s, 0.0)
                s = np.sort(s)
                
                # Ensure w is non-negative and sums to 1
                w = np.maximum(w, 0.0)
                w_sum = np.sum(w)
                if w_sum < 1e-9:
                    return 1.0
                w = w / w_sum
                
                return -calc_P_efficient(s, w)
            except:
                return 1.0
        
        # Start from best solution
        s0 = best_support.copy()
        w0 = best_weights.copy()
        
        # Normalize w0
        w0 = w0 / np.sum(w0)
        
        x0 = np.concatenate([s0, w0])
        
        # Use Nelder-Mead
        try:
            res = minimize(neg_P, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < -best_P:
                s_opt = res.x[:n_best]
                w_opt = res.x[n_best:]
                s_opt = np.maximum(s_opt, 0.0)
                s_opt = np.sort(s_opt)
                w_opt = np.maximum(w_opt, 0.0)
                w_opt = w_opt / np.sum(w_opt)
                try_solution(s_opt, w_opt)
        except:
            pass
        
        # Also try with fixed support, optimize weights only
        if time.time() < deadline - 10:
            def neg_P_w(params):
                try:
                    w = np.maximum(params, 0.0)
                    w_sum = np.sum(w)
                    if w_sum < 1e-9:
                        return 1.0
                    w = w / w_sum
                    return -calc_P_efficient(best_support, w)
                except:
                    return 1.0
            
            try:
                res = minimize(neg_P_w, w0.copy(), method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < -best_P:
                    w_opt = res.x
                    w_opt = np.maximum(w_opt, 0.0)
                    w_opt = w_opt / np.sum(w_opt)
                    try_solution(best_support, w_opt)
            except:
                pass
    
    # Final validation and return
    if best_support is None or best_weights is None:
        # Fallback to seed
        return support_seed, weights_seed
    
    s_final = np.asarray(best_support, dtype=np.float64).flatten()
    w_final = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Final checks
    if len(s_final) != len(w_final) or len(s_final) < 1:
        return support_seed, weights_seed
    if np.any(np.isnan(s_final)) or np.any(np.isnan(w_final)):
        return support_seed, weights_seed
    if np.any(np.isinf(s_final)) or np.any(np.isinf(w_final)):
        return support_seed, weights_seed
    if np.any(s_final < -1e-9):
        return support_seed, weights_seed
    if np.any(w_final < -1e-9):
        return support_seed, weights_seed
    if np.sum(w_final) <= 1e-9:
        return support_seed, weights_seed
    
    return s_final, w_final
