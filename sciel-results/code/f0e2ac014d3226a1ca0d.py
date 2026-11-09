import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    
    Strategy:
    1. Exploit the problem structure: M is dominated by H(x) (row sums) and H(x+y) (diagonals).
       To maximize D/M, we should minimize M (make marginals as concentrated as possible) 
       while maximizing D (spread out x-y).
    2. However, we cannot make marginals too concentrated without hurting D or violating 
       non-negativity. A balanced approach is to use a parameterized family that allows 
       controlling the "spread" of rows/cols and the "bias" towards anti-diagonals.
    3. Parameterization:
       P[y, x] ∝ exp(α * (x - y) + β * x + γ * y)
       This is a rank-1 exponential family. 
       - α controls the anti-diagonal bias (affects D).
       - β and γ control row/col biases (affect M).
       Since P is normalized, adding a constant to all exponents doesn't change P.
       We can fix one parameter or use symmetric/asymmetric forms.
       
       Actually, a more flexible low-dim parameterization:
       P[y, x] ∝ exp( u * (x-y) + v * (x+y) ) 
       This covers the main structural features.
       
       Let's try a slightly richer family to allow asymmetry:
       P[y, x] ∝ exp( a*(x-y) + b*x + c*y )
       
       We will optimize (a, b, c) using L-BFGS-B or Nelder-Mead.
       Then, we will also try a multi-start approach with random perturbations 
       and potentially a second-order correction or direct coordinate ascent 
       on the full grid if time permits, starting from the best structured solution.
    """
    n = 10
    N = n * n
    start_time = time.time()
    deadline = start_time + 610  # 20s buffer
    
    # Pre-compute indices
    ys = np.repeat(np.arange(n), n)
    xs = np.tile(np.arange(n), n)
    
    idx_dm1 = (xs - ys + n - 1).astype(np.int32)
    idx_x   = xs.astype(np.int32)
    idx_d1  = (xs + ys).astype(np.int32)
    idx_d2  = (xs + 2 * ys).astype(np.int32)
    
    bin_sizes = {
        'dm1': 2 * n - 1,
        'x': n,
        'd1': 2 * n - 1,
        'd2': 3 * n - 1
    }

    def entropy_from_counts(counts):
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
        Objective function for optimization.
        params: [a, b, c] for P ∝ exp(a*(x-y) + b*x + c*y)
        """
        a, b, c = params
        # Exponents
        exps = a * (xs - ys) + b * xs + c * ys
        # Softmax
        exps_shifted = exps - np.max(exps)
        p = np.exp(exps_shifted)
        s = p.sum()
        if s < 1e-15:
            return 1e9
        p = p / s
        
        score = evaluate_P(p)
        if not np.isfinite(score):
            return 1e9
        return -score

    # --- Phase 1: Optimize the structured low-dimensional family ---
    # Bounds for a, b, c
    # a: anti-diagonal bias. If a is large positive, mass goes to x-y=9 (top-right). 
    #    If a is large negative, mass goes to x-y=-9 (bottom-left).
    #    D is max when spread is uniform. But M is min when marginals are sharp.
    #    There is a trade-off.
    #    Let's search a reasonable range.
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    best_score = -np.inf
    best_params = None
    best_P = None
    
    # Multi-start L-BFGS-B
    # Use a few random starts and some structured starts
    starts = [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 1],
        [-1, -1, -1],
        [2, -1, 0],
        [-2, 1, 0],
    ]
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(20):
        starts.append(np.random.uniform(-3, 3, 3))
    
    for start in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(score_from_params, start, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
            if res.fun < 1e8: # Valid score
                cur_score = -res.fun
                if cur_score > best_score:
                    best_score = cur_score
                    best_params = res.x
                    # Reconstruct P
                    a, b, c = res.x
                    exps = a * (xs - ys) + b * xs + c * ys
                    exps_shifted = exps - np.max(exps)
                    p = np.exp(exps_shifted)
                    p /= p.sum()
                    best_P = p
        except Exception:
            pass

    if best_P is None:
        # Fallback to uniform
        best_P = np.ones(N) / N
        best_score = evaluate_P(best_P)

    # --- Phase 2: Local refinement on the full 100-dim space ---
    # The structured family might be a local optimum. 
    # We can try to improve it by allowing arbitrary deviations from the structured form.
    # Parameterize P = exp(log_best_P + delta) where delta is small?
    # Or just run L-BFGS-B on log-probs starting from the best structured P.
    
    if time.time() < deadline - 10:
        log_P = np.log(best_P + 1e-15)
        
        def score_from_log_probs(log_p):
            p = np.exp(log_p - np.max(log_p))
            p /= p.sum()
            score = evaluate_P(p)
            if not np.isfinite(score):
                return 1e9
            return -score

        # Use a few random perturbations of the starting point
        best_log_p = log_P
        best_local_score = best_score
        
        # Try a few random restarts in the neighborhood
        for i in range(10):
            if time.time() > deadline - 5:
                break
            # Perturb log_P slightly
            perturbation = np.random.normal(0, 0.1, N)
            start_log_p = log_P + perturbation
            
            try:
                res = minimize(score_from_log_probs, start_log_p, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
                if res.fun < 1e8:
                    cur_score = -res.fun
                    if cur_score > best_local_score:
                        best_local_score = cur_score
                        best_log_p = res.x
            except Exception:
                pass
        
        if best_local_score > best_score:
            best_score = best_local_score
            p = np.exp(best_log_p - np.max(best_log_p))
            p /= p.sum()
            best_P = p

    # Final check and return
    P_final = np.clip(best_P, 0, None)
    s = P_final.sum()
    if s < 1e-9:
        P_final = np.ones(N) / N
    else:
        P_final /= s
        
    P_final = P_final.reshape((n, n))
    
    if not np.all(np.isfinite(P_final)) or P_final.shape != (n, n):
        return np.ones((n, n)) / N
        
    return P_final

if __name__ == "__main__":
    pass
