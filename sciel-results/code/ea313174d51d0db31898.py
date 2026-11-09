# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio H(X-Y) / max(H(X), H(X+Y), H(X+2Y)) for a distribution on a 10x10 grid.
    """
    start_time = time.time()
    deadline = start_time + 500  # 500s safety margin
    
    n = 10
    # Precompute indices for efficiency
    x_vals = np.arange(n)
    y_vals = np.arange(n)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy') # X[y, x], Y[y, x]
    # Note: P[y, x] corresponds to Y=y, X=x
    # So X grid is x_vals repeated along y, Y grid is y_vals repeated along x
    # X[y, x] = x, Y[y, x] = y
    
    diffs = X - Y       # X - Y
    sums0 = X           # X
    sums1 = X + Y       # X + Y
    sums2 = X + 2*Y     # X + 2*Y

    def compute_entropies(P):
        # P is flattened 100 or 2D 10x10
        if P.ndim == 1:
            P = P.reshape((n, n))
        
        # Clip and normalize to ensure valid probability distribution for entropy calc
        P_clean = np.clip(P, 0, None)
        s = P_clean.sum()
        if s <= 1e-12:
            return 0.0, 0.0, 0.0, 0.0
        
        P_norm = P_clean / s
        
        def shannon(dist):
            # dist is a vector of probabilities (not normalized yet? No, let's normalize inside)
            d = np.clip(dist, 0, None)
            s_d = d.sum()
            if s_d <= 1e-12:
                return 0.0
            d_norm = d / s_d
            # Ignore zeros
            mask = d_norm > 1e-12
            if not np.any(mask):
                return 0.0
            return -np.sum(d_norm[mask] * np.log2(d_norm[mask]))
        
        # Pool probabilities for each projection value
        # We need sum of P[y,x] for all (y,x) such that val == k
        
        # Efficient pooling using np.bincount or similar?
        # Values are integers in small ranges.
        
        # X-Y: range -9 to 9. Offset by 9 to get 0..18
        d_vals = (diffs + 9).astype(int).flatten()
        p_flat = P_norm.flatten()
        h_diff = shannon(np.bincount(d_vals, weights=p_flat, minlength=19))
        
        # X: range 0 to 9
        x_vals_flat = X.flatten().astype(int)
        h_x = shannon(np.bincount(x_vals_flat, weights=p_flat, minlength=10))
        
        # X+Y: range 0 to 18
        s1_vals = sums1.flatten().astype(int)
        h_sum1 = shannon(np.bincount(s1_vals, weights=p_flat, minlength=19))
        
        # X+2Y: range 0 to 27
        s2_vals = sums2.flatten().astype(int)
        h_sum2 = shannon(np.bincount(s2_vals, weights=p_flat, minlength=28))
        
        return h_diff, h_x, h_sum1, h_sum2

    def objective(params):
        # params: [alpha, beta]
        alpha, beta = params
        # Model: exp(-alpha * (x+y-9)^2 - beta * (x-y)^2)
        # Center at 4.5, 4.5
        x_c = X - 4.5
        y_c = Y - 4.5
        # x+y-9 = (x-4.5) + (y-4.5)
        # x-y = (x-4.5) - (y-4.5)
        s_c = x_c + y_c
        d_c = x_c - y_c
        
        log_P = -alpha * (s_c ** 2) - beta * (d_c ** 2)
        
        # Normalize
        log_P -= np.max(log_P)
        P = np.exp(log_P)
        P_sum = P.sum()
        if P_sum < 1e-12:
            return 1e6 # Invalid
        
        P_norm = P / P_sum
        
        h_diff, h_x, h_sum1, h_sum2 = compute_entropies(P_norm)
        
        M = max(h_x, h_sum1, h_sum2)
        if M < 1e-9:
            return 1e6
        
        ratio = h_diff / M
        return -np.log(ratio) # Minimize negative log ratio

    best_P = np.ones((n, n)) / (n*n)
    best_score = -np.inf
    best_val = 0.0

    # 1. Parametric Optimization
    # Grid search for initial alpha, beta
    # alpha controls concentration of sum (want high)
    # beta controls concentration of diff (want low)
    
    # Try a range of alphas and betas
    alphas = np.linspace(0.5, 15.0, 30)
    betas = np.linspace(0.0, 5.0, 20)
    
    # Coarse grid search
    for a in alphas:
        for b in betas:
            if time.time() > deadline:
                break
            val = objective([a, b])
            if val < best_score:
                best_score = val
                best_params = [a, b]
        
        if time.time() > deadline:
            break
    
    if best_score != -np.inf:
        # Refine with Nelder-Mead
        res = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
        
        # Calculate best P from best params
        alpha, beta = best_params
        x_c = X - 4.5
        y_c = Y - 4.5
        s_c = x_c + y_c
        d_c = x_c - y_c
        log_P = -alpha * (s_c ** 2) - beta * (d_c ** 2)
        log_P -= np.max(log_P)
        best_P = np.exp(log_P)
        best_P /= best_P.sum()
        
        h_diff, h_x, h_sum1, h_sum2 = compute_entropies(best_P)
        M = max(h_x, h_sum1, h_sum2)
        if M > 1e-9:
            best_val = h_diff / M

    # 2. Local Refinement on full 100 params if time permits
    if time.time() < deadline - 100: # Need 100s for NM on 100 dims? Maybe less.
        # Initialize with best_P
        P_flat = best_P.flatten().copy()
        
        def full_objective(params):
            P = params.reshape((n, n))
            P_clean = np.clip(P, 0, None)
            s = P_clean.sum()
            if s < 1e-9:
                return 1e6
            P_norm = P_clean / s
            h_diff, h_x, h_sum1, h_sum2 = compute_entropies(P_norm)
            M = max(h_x, h_sum1, h_sum2)
            if M < 1e-9:
                return 1e6
            ratio = h_diff / M
            return -np.log(ratio)
        
        # Use Powell or NM
        # NM is stochastic, good for non-differentiable.
        try:
            res_full = minimize(full_objective, P_flat, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res_full.fun < best_score:
                P_opt = res_full.x.reshape((n, n))
                P_opt = np.clip(P_opt, 0, None)
                P_opt /= P_opt.sum()
                best_P = P_opt
                
                h_diff, h_x, h_sum1, h_sum2 = compute_entropies(best_P)
                M = max(h_x, h_sum1, h_sum2)
                if M > 1e-9:
                    val = h_diff / M
                    if val > best_val:
                        best_val = val
        except Exception:
            pass

    # Final validation and cleanup
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P = best_P / s
    else:
        best_P = np.ones((n, n)) / (n*n)
        
    return best_P

# EVOLVE-BLOCK-END
