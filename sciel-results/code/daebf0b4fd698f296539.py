import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    
    Strategy:
    1. Recognize that the objective depends only on the marginals of linear forms.
    2. Parameterize P using low-dimensional structures (diagonal weights, row weights, 
       or a small number of "atoms") to reduce the effective search space.
    3. Use a hybrid approach: 
       a. Coarse grid search over structural parameters.
       b. Local refinement using L-BFGS-B on the structural parameters.
    4. This avoids the 100-dimensional flatness of the simplex where gradient 
       methods struggle, focusing instead on the geometry of the marginals.
    """
    n = 10
    N = n * n
    start_time = time.time()
    deadline = start_time + 620  # 10s safety buffer

    # Pre-compute indices for entropy calculation
    ys = np.repeat(np.arange(n), n)
    xs = np.tile(np.arange(n), n)
    
    # Indices for binning
    idx_dm1 = (xs - ys + n - 1).astype(np.int32)       # x - y + 9 -> 0..18
    idx_x   = xs.astype(np.int32)                      # x -> 0..9
    idx_d1  = (xs + ys).astype(np.int32)               # x + y -> 0..18
    idx_d2  = (xs + 2 * ys).astype(np.int32)           # x + 2y -> 0..27
    
    bin_sizes = {
        'dm1': 2 * n - 1,
        'x': n,
        'd1': 2 * n - 1,
        'd2': 3 * n - 1
    }

    def entropy_from_counts(counts):
        """Calculate Shannon entropy (bits) from a count/probability vector."""
        mask = counts > 1e-15
        if not np.any(mask):
            return 0.0
        p = counts[mask]
        p_sum = p.sum()
        if p_sum < 1e-15:
            return 0.0
        p = p / p_sum
        return float(-np.sum(p * np.log2(p)))

    def evaluate_P(P_flat):
        """
        Evaluate the score for a flattened probability vector P_flat.
        """
        if not np.all(np.isfinite(P_flat)):
            return -np.inf
        
        P_norm = np.clip(P_flat, 0, None)
        s = P_norm.sum()
        if s < 1e-9:
            return -np.inf
        P_norm /= s

        c_dm1 = np.bincount(idx_dm1, weights=P_norm, minlength=bin_sizes['dm1'])
        c_x   = np.bincount(idx_x,   weights=P_norm, minlength=bin_sizes['x'])
        c_d1  = np.bincount(idx_d1,  weights=P_norm, minlength=bin_sizes['d1'])
        c_d2  = np.bincount(idx_d2,  weights=P_norm, minlength=bin_sizes['d2'])

        H_dm1 = entropy_from_counts(c_dm1)
        H_x   = entropy_from_counts(c_x)
        H_d1  = entropy_from_counts(c_d1)
        H_d2  = entropy_from_counts(c_d2)

        D = H_dm1
        M = max(H_x, H_d1, H_d2)
        
        if M < 1e-9:
            return -np.inf
            
        return D / M

    def score_from_params(params):
        """
        Objective function for optimization on structural parameters.
        params: [log_weights for diagonals] or [log_weights for rows/cols] etc.
        We use a flexible parameterization:
        Type 0: Diagonal weights (19 params)
        Type 1: Row weights (10 params)
        Type 2: Column weights (10 params)
        Type 3: Mixed (Row + Col + Diag)
        
        For simplicity and speed, we'll stick to a few key structural families.
        """
        if time.time() > deadline:
            return 1e9

        # Determine which parameterization to use based on length or a flag
        # Here we assume params are log-weights for a specific structure.
        # We'll implement a few specific parameterizations and select the best.
        
        # To keep it simple within the function signature, we'll use a specific 
        # parameterization: Weights on the 19 diagonals (x-y).
        # This directly controls D. We need to see how it affects M.
        
        # Let's try a more general approach: 
        # P[y, x] = A[y] * B[x] * C[x-y] ? No, that's too complex.
        
        # Let's use the "Diagonal Weight" parameterization.
        # There are 19 diagonals. Let w_d be the weight for diagonal d (where d = x-y+9).
        # We distribute w_d equally among the cells on that diagonal.
        
        if len(params) == 19:
            log_w = params
            log_w_shifted = log_w - np.max(log_w)
            w = np.exp(log_w_shifted)
            w_sum = w.sum()
            if w_sum < 1e-15:
                return 1e9
            w = w / w_sum
            
            P = np.zeros(N)
            # Map each cell to its diagonal index
            for i in range(n):
                for j in range(n):
                    d = j - i + n - 1
                    P[i*n + j] = w[d] / (n - abs(j-i)) # Uniform on the diagonal
            
            score = evaluate_P(P)
            if not np.isfinite(score):
                return 1e9
            return -score
        
        # If we want to support other parameterizations, we'd add more branches.
        # For now, let's also try Row-Column product: P[y,x] = R[y] * C[x]
        # This is 20 params.
        elif len(params) == 20:
            log_r = params[:10]
            log_c = params[10:]
            log_r_shifted = log_r - np.max(log_r)
            log_c_shifted = log_c - np.max(log_c)
            r = np.exp(log_r_shifted)
            c = np.exp(log_c_shifted)
            r_sum = r.sum()
            c_sum = c.sum()
            if r_sum < 1e-15 or c_sum < 1e-15:
                return 1e9
            r = r / r_sum
            c = c / c_sum
            
            P = np.outer(r, c).flatten()
            score = evaluate_P(P)
            if not np.isfinite(score):
                return 1e9
            return -score

        return 1e9

    best_score = -np.inf
    best_P = np.ones((n, n)) / N # Fallback uniform

    # --- Phase 1: Coarse Grid Search on Structural Parameters ---
    
    # Strategy A: Diagonal Weights
    # We want to maximize D (entropy of x-y). Uniform diagonals give max D.
    # But M (max of x, x+y, x+2y) also changes.
    # Let's sample diagonal weight vectors.
    
    # 1. Uniform diagonals
    init_diags = np.zeros(19)
    res = minimize(score_from_params, init_diags, method='Nelder-Mead', 
                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
    if res.fun < 1e8:
        score = -res.fun
        if score > best_score:
            best_score = score
            # Reconstruct P
            log_w = res.x
            log_w_shifted = log_w - np.max(log_w)
            w = np.exp(log_w_shifted)
            w = w / w.sum()
            P = np.zeros(N)
            for i in range(n):
                for j in range(n):
                    d = j - i + n - 1
                    P[i*n + j] = w[d] / (n - abs(j-i))
            best_P = P.reshape((n, n))

    # 2. Skewed diagonals (favor center or edges)
    for skew in [0.5, 1.0, 2.0, -0.5, -1.0, -2.0]:
        init = np.arange(19) * skew
        res = minimize(score_from_params, init, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < 1e8:
            score = -res.fun
            if score > best_score:
                best_score = score
                log_w = res.x
                log_w_shifted = log_w - np.max(log_w)
                w = np.exp(log_w_shifted)
                w = w / w.sum()
                P = np.zeros(N)
                for i in range(n):
                    for j in range(n):
                        d = j - i + n - 1
                        P[i*n + j] = w[d] / (n - abs(j-i))
                best_P = P.reshape((n, n))

    # Strategy B: Row-Column Product P[y,x] = R[y]C[x]
    # This allows control over marginals of x and y separately.
    
    # 1. Uniform rows/cols
    init_rc = np.zeros(20)
    res = minimize(score_from_params, init_rc, method='Nelder-Mead', 
                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
    if res.fun < 1e8:
        score = -res.fun
        if score > best_score:
            best_score = score
            log_r = res.x[:10]
            log_c = res.x[10:]
            log_r_shifted = log_r - np.max(log_r)
            log_c_shifted = log_c - np.max(log_c)
            r = np.exp(log_r_shifted)
            c = np.exp(log_c_shifted)
            r = r / r.sum()
            c = c / c.sum()
            P = np.outer(r, c).flatten()
            best_P = P.reshape((n, n))

    # 2. Skewed rows/cols
    for skew_r in [0.5, -0.5]:
        for skew_c in [0.5, -0.5]:
            init = np.zeros(20)
            init[:10] = np.arange(10) * skew_r
            init[10:] = np.arange(10) * skew_c
            res = minimize(score_from_params, init, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < 1e8:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    log_r = res.x[:10]
                    log_c = res.x[10:]
                    log_r_shifted = log_r - np.max(log_r)
                    log_c_shifted = log_c - np.max(log_c)
                    r = np.exp(log_r_shifted)
                    c = np.exp(log_c_shifted)
                    r = r / r.sum()
                    c = c / c.sum()
                    P = np.outer(r, c).flatten()
                    best_P = P.reshape((n, n))

    # --- Phase 2: Local Refinement on Full Simplex from Best Structural Result ---
    # Use the best structural P as a starting point for full 100-dim L-BFGS-B.
    # This combines the global structural insight with local fine-tuning.
    
    if best_score > -np.inf:
        init_log_p = np.log(best_P.flatten() + 1e-10)
        
        def score_from_log_probs(log_p):
            if time.time() > deadline:
                return 1e9
            log_p_shifted = log_p - np.max(log_p)
            p = np.exp(log_p_shifted)
            s = p.sum()
            if s < 1e-15:
                return 1e9
            p = p / s
            score = evaluate_P(p)
            if not np.isfinite(score):
                return 1e9
            return -score

        try:
            res = minimize(score_from_log_probs, init_log_p, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
            if res.fun < 1e8:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    log_p_shifted = res.x - np.max(res.x)
                    p = np.exp(log_p_shifted)
                    p = p / p.sum()
                    best_P = p.reshape((n, n))
        except Exception:
            pass

    # Final safety check
    if not np.all(np.isfinite(best_P)) or best_P.shape != (n, n):
        return np.ones((n, n)) / N
        
    # Clip and normalize for final output
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-9:
        best_P = best_P / s
    else:
        return np.ones((n, n)) / N

    return best_P

if __name__ == "__main__":
    pass
