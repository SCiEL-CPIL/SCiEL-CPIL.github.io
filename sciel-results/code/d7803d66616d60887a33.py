import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    
    Strategy:
    1. Parameterize the 10x10 probability matrix P using a low-dimensional 
       structured model: P[y, x] = exp(a_y + b_x). This reduces the search 
       space from 100 free parameters (on the simplex) to 20 parameters 
       (10 for rows, 10 for columns), preserving the additive structure 
       relevant to the entropy terms (x, y, x+y, x-y).
    2. Use multi-start gradient ascent (L-BFGS-B) on the log-parameters 
       (a_y, b_x) to maximize the ratio D/M.
    3. The structured initialization and low-dimensional search allow for 
       rapid convergence and avoid the pitfalls of high-dimensional stochastic 
       search.
    """
    n = 10
    N = n * n
    start_time = time.time()
    deadline = start_time + 590  # 10s buffer for overhead

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
        
        # Clip and normalize
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
        Objective function for minimization.
        Input: params (20 values: 10 for row weights, 10 for col weights)
        Output: Negative score
        """
        a_y = params[:n]
        b_x = params[n:]
        
        # Construct P[y, x] = exp(a_y + b_x)
        # Shift to prevent overflow/underflow
        log_p = a_y[:, None] + b_x[None, :]
        log_p_shifted = log_p - np.max(log_p)
        p = np.exp(log_p_shifted)
        s = p.sum()
        if s < 1e-15:
            return 1e9 
        p = p / s
        
        score = evaluate_P(p.flatten())
        if not np.isfinite(score):
            return 1e9
        return -score

    # --- Multi-Start Gradient Ascent ---
    # We will try multiple structured initializations and refine with L-BFGS-B
    
    best_score = -np.inf
    best_params = None
    
    # Define diverse initializations
    inits = []
    
    # 1. Uniform (all zeros)
    inits.append(np.zeros(2*n))
    
    # 2. Diagonal bias (a_y + b_x high when y==x)
    init_diag = np.zeros(2*n)
    for i in range(n):
        init_diag[i] = 1.0       # a_y
        init_diag[n+i] = 1.0     # b_x
    # This makes exp(2) for diagonal, exp(0) for off-diagonal
    inits.append(init_diag)
    
    # 3. Anti-diagonal bias (a_y + b_x high when y+x=9)
    init_anti = np.zeros(2*n)
    for i in range(n):
        init_anti[i] = 1.0
        init_anti[n+(n-1-i)] = 1.0
    inits.append(init_anti)
    
    # 4. Row bias (a_y high, b_x low)
    init_row = np.zeros(2*n)
    init_row[:n] = 1.0
    inits.append(init_row)
    
    # 5. Col bias (a_y low, b_x high)
    init_col = np.zeros(2*n)
    init_col[n:] = 1.0
    inits.append(init_col)
    
    # 6. Random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        inits.append(rng.normal(0, 1, 2*n))
        
    # 7. Structured "checkerboard"
    init_check = np.zeros(2*n)
    for i in range(n):
        if i % 2 == 0:
            init_check[i] = 1.0
            init_check[n+i] = 1.0
        else:
            init_check[i] = -1.0
            init_check[n+i] = -1.0
    inits.append(init_check)

    bounds = [(-10, 10)] * (2*n)
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        try:
            result = minimize(
                score_from_params,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            curr_score = -result.fun
            if curr_score > best_score:
                best_score = curr_score
                best_params = result.x
                
        except Exception as e:
            pass

    # --- Final Conversion and Validation ---
    if best_params is None:
        # Fallback to uniform
        P_final = np.ones((n, n)) / N
    else:
        a_y = best_params[:n]
        b_x = best_params[n:]
        log_p = a_y[:, None] + b_x[None, :]
        log_p_shifted = log_p - np.max(log_p)
        P_final = np.exp(log_p_shifted)
        P_final = np.clip(P_final, 0, None)
        s = P_final.sum()
        if s > 1e-9:
            P_final /= s
        else:
            P_final = np.ones((n, n)) / N
            
    # Final safety check
    if not np.all(np.isfinite(P_final)) or P_final.shape != (n, n):
        return np.ones((n, n)) / N
        
    return P_final

if __name__ == "__main__":
    pass
