# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Maximizes D/M where D = H_{-1} (entropy of x-y distribution) and
    M = max(H_0, H_1, H_2) (max entropy of x, x+y, x+2y distributions).
    
    Returns:
        P: np.ndarray of shape (10, 10), non-negative, summing to 1.
    """
    n = 10
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    # Precompute projection indices
    x_idx = np.arange(n)
    y_idx = np.arange(n)
    
    # For each slope r, compute the value x + r*y for each cell (y, x)
    # P[y, x] corresponds to X=x, Y=y
    # So value = x + r*y
    
    def compute_entropies(P):
        """Compute H_{-1}, H_0, H_1, H_2 from distribution P[y, x]."""
        # P has shape (n, n) where P[y, x] is probability of (X=x, Y=y)
        
        # H_{-1}: distribution of x - y
        # Values range from -9 to 9
        diff_vals = x_idx[None, :] - y_idx[:, None]  # shape (n, n), [y, x]
        
        # H_0: distribution of x (values 0-9)
        x_vals = x_idx[None, :]  # shape (1, n) -> broadcast
        
        # H_1: distribution of x + y
        sum_vals = x_idx[None, :] + y_idx[:, None]
        
        # H_2: distribution of x + 2y
        sum2_vals = x_idx[None, :] + 2 * y_idx[:, None]
        
        def shannon_entropy_from_values(P, vals):
            """Compute Shannon entropy given probability matrix and value matrix."""
            # Pool by value
            unique_vals = np.unique(vals)
            probs = []
            for v in unique_vals:
                mask = vals == v
                p = P[mask].sum()
                if p > 1e-15:
                    probs.append(p)
            probs = np.array(probs)
            if len(probs) == 0:
                return 0.0
            # Shannon entropy in bits
            return -np.sum(probs * np.log2(probs))
        
        H_neg1 = shannon_entropy_from_values(P, diff_vals)
        H_0 = shannon_entropy_from_values(P, x_vals)
        H_1 = shannon_entropy_from_values(P, sum_vals)
        H_2 = shannon_entropy_from_values(P, sum2_vals)
        
        return H_neg1, H_0, H_1, H_2
    
    def objective_and_grad(params):
        """
        Objective: minimize -D/M (i.e., maximize D/M)
        params: logit parameters for P (shape 100)
        """
        # Convert logits to probabilities via softmax
        # Use a stable softmax
        params_clipped = np.clip(params, -50, 50)
        exp_params = np.exp(params_clipped - np.max(params_clipped))
        P = exp_params / exp_params.sum()
        P = P.reshape(n, n)
        
        H_neg1, H_0, H_1, H_2 = compute_entropies(P)
        M = max(H_0, H_1, H_2)
        
        if M < 1e-15:
            return 1e10, None
        
        ratio = H_neg1 / M
        return -ratio, None
    
    def objective_only(params):
        val, _ = objective_and_grad(params)
        return val
    
    # Try multiple initializations
    best_P = None
    best_score = -np.inf
    
    # Initializations to try
    inits = []
    
    # 1. Uniform
    inits.append(np.zeros(n * n))
    
    # 2. Diagonal concentration (x = y)
    diag = np.zeros(n * n)
    for i in range(n):
        diag[i * n + i] = 5.0
    inits.append(diag)
    
    # 3. Anti-diagonal concentration (x + y = 9)
    anti = np.zeros(n * n)
    for i in range(n):
        anti[i * n + (n - 1 - i)] = 5.0
    inits.append(anti)
    
    # 4. Corner concentrations
    corner1 = np.zeros(n * n)
    corner1[0] = 5.0  # (y=0, x=0)
    inits.append(corner1)
    
    corner2 = np.zeros(n * n)
    corner2[n * n - 1] = 5.0  # (y=9, x=9)
    inits.append(corner2)
    
    # 5. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(rng.normal(0, 2, n * n))
    
    # 6. Concentrate on cells where x-y has many distinct values but x, x+y, x+2y are spread
    # Actually, we want x-y to have HIGH entropy (many distinct values with balanced probs)
    # and x, x+y, x+2y to have LOW entropy (concentrated)
    
    # Let's think: x-y ranges from -9 to 9 (19 values)
    # x ranges from 0 to 9 (10 values)
    # x+y ranges from 0 to 18 (19 values)
    # x+2y ranges from 0 to 27 (28 values)
    
    # To maximize H(x-y), we want mass spread across many x-y values
    # To minimize max(H(x), H(x+y), H(x+2y)), we want mass concentrated in few values of one of these
    
    # Strategy: concentrate on a line where x is constant (so H(x) = 0)
    # If x = c for all mass, then H(x) = 0
    # x-y = c-y, so as y varies, x-y takes all values c-9 to c-0, i.e., 10 distinct values
    # x+y = c+y, 10 distinct values
    # x+2y = c+2y, 10 distinct values
    # So M = max(0, H(x+y), H(x+2y))
    # H(x-y) with uniform y gives H = log2(10) ≈ 3.32
    # H(x+y) with uniform y also gives H = log2(10) ≈ 3.32
    # So ratio ≈ 1
    
    # Better: concentrate on cells where x-y takes many values but x+y takes few
    # If we concentrate on a diagonal x - y = c, then H(x-y) = 0 (bad)
    # If we concentrate on a vertical line x = c, H(x) = 0, good for denominator
    # But H(x-y) and H(x+y) are both log2(10) if y is uniform
    
    # What if we use a non-uniform distribution on a vertical line?
    # x = c, y ~ distribution p(y)
    # H(x-y) = H(c-y) = H(y) (since c-y is just a shift)
    # H(x+y) = H(c+y) = H(y)
    # H(x+2y) = H(c+2y)
    # If y is uniform on 0..9, H(y) = log2(10)
    # H(x+2y) with y uniform: values c, c+2, c+4, ..., c+18, all distinct, so H = log2(10)
    # So M = log2(10), D = log2(10), ratio = 1
    
    # To do better, we need D > M
    # D = H(x-y), M = max(H(x), H(x+y), H(x+2y))
    
    # Consider putting mass on multiple vertical lines
    # Let's say mass on x=0 and x=9 only
    # Then H(x) = 1 bit (if equal mass)
    # x-y: for x=0, values -y; for x=9, values 9-y
    # If y uniform on both, x-y takes values -9..-1 (from x=0) and 0..9 (from x=9)
    # That's 19 distinct values, each with prob 1/100
    # H(x-y) = log2(19) ≈ 4.25
    # x+y: for x=0, y; for x=9, 9+y
    # Values 0..9 (from x=0) and 9..18 (from x=9)
    # Value 9 has prob 2/100, others have 1/100
    # H(x+y) = -[18*(1/100)*log2(1/100) + (2/100)*log2(2/100)]
    # = -[18*0.01*(-6.644) + 0.02*(-5.644)]
    # = -[-1.196 - 0.113] = 1.309
    # x+2y: for x=0, 2y; for x=9, 9+2y
    # Values 0,2,4,...,18 (from x=0) and 9,11,13,...,27 (from x=9)
    # All distinct? 0,2,4,6,8,10,12,14,16,18 and 9,11,13,15,17,19,21,23,25,27
    # Yes, all 20 distinct values
    # H(x+2y) = log2(20) ≈ 4.32
    
    # So M = max(1, 1.309, 4.32) = 4.32
    # D = 4.25
    # Ratio = 4.25/4.32 ≈ 0.98 < 1
    
    # Hmm, that's worse. The issue is x+2y has too many distinct values.
    
    # What if we concentrate on horizontal lines? y = c
    # Then H(y) isn't directly used, but:
    # x-y = x-c, so H(x-y) = H(x)
    # x = x, H(x)
    # x+y = x+c, H(x+y) = H(x)
    # x+2y = x+2c, H(x+2y) = H(x)
    # So D = H(x), M = H(x), ratio = 1
    
    # Let me try a different approach: use optimization to find a good P
    
    # Use L-BFGS-B with constraints
    def optimize_from_init(init_logits):
        """Run L-BFGS-B optimization from given initial logits."""
        try:
            result = opt.minimize(
                objective_only,
                init_logits,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if result.success or result.fun < 0:
                params_opt = result.x
                exp_params = np.exp(np.clip(params_opt, -50, 50) - np.max(np.clip(params_opt, -50, 50)))
                P_opt = (exp_params / exp_params.sum()).reshape(n, n)
                H_neg1, H_0, H_1, H_2 = compute_entropies(P_opt)
                M = max(H_0, H_1, H_2)
                if M > 1e-15:
                    score = H_neg1 / M
                    return P_opt, score
        except Exception:
            pass
        return None, -np.inf
    
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        P_opt, score = optimize_from_init(init)
        if P_opt is not None and score > best_score:
            best_score = score
            best_P = P_opt
    
    # If no good result, fall back to uniform
    if best_P is None:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Clip and normalize
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total > 1e-15:
        best_P = best_P / total
    else:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    return best_P

# EVOLVE-BLOCK-END
