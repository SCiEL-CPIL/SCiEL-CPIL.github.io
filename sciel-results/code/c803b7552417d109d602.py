# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Precompute projection indices for x, y, x-y, x+y, x+2y.
    2. Use Differential Evolution (global optimization) to search the 
       100-dimensional space of log-probabilities. This helps escape 
       local maxima that trap gradient-based methods.
    3. Polish the best result found by DE using L-BFGS-B for high precision.
    """
    n = 10
    deadline = time.time() + 590.0  # Safety margin (limit is 630s)
    
    # Grid coordinates
    # P[y, x] -> y is row, x is col
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Flatten for binning
    # Note: The problem defines P[y, x]. 
    d_val = (xs - ys).flatten()
    m0_val = xs.flatten()
    m1_val = (xs + ys).flatten()
    m2_val = (xs + 2 * ys).flatten()
    
    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)
    
    def compute_entropies_from_p(p_flat):
        """
        p_flat: shape (100,) normalized probabilities
        Returns: D, M
        """
        # Use bincount with weights for efficiency
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)
        
        def shannon(p_dist):
            p_nonzero = p_dist[p_dist > 1e-15]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def objective(log_p_params):
        """
        log_p_params: shape (100,) log-probabilities (unnormalized)
        Returns: negative ratio -D/M (to be minimized)
        """
        if time.time() > deadline:
            return 1e9
            
        # Softmax to get probabilities
        # Subtract max for numerical stability
        log_p_flat = log_p_params - np.max(log_p_params)
        p_flat = np.exp(log_p_flat)
        sum_p = np.sum(p_flat)
        
        if sum_p < 1e-15:
            return 1e9
            
        p_flat /= sum_p
        
        H_d, M = compute_entropies_from_p(p_flat)
        
        if M < 1e-9:
            return 1e9
            
        return -H_d / M

    # Initial best: Uniform distribution
    # Uniform corresponds to all log_p = 0
    init_params = np.zeros(100)
    best_score = -objective(init_params)
    best_params = init_params.copy()
    
    # Phase 1: Global Optimization with Differential Evolution
    # DE is robust to local minima. We define bounds for log_p.
    # Since softmax is invariant to shifts, we can fix one variable or just use wide bounds.
    # Using bounds [-10, 10] is sufficient to explore diverse distributions.
    
    bounds = [(-10.0, 10.0) for _ in range(100)]
    
    try:
        # maxiter and popsize control the search depth.
        # We limit the time by checking inside the objective, but DE has its own termination.
        # We set a reasonable maxiter to ensure it doesn't run forever if it converges slowly.
        res_de = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False  # We will polish manually to control time
        )
        
        if res_de.fun < -best_score:
            best_score = -res_de.fun
            best_params = res_de.x
            
    except Exception as e:
        # If DE fails, we continue with the initial best
        pass

    # Phase 2: Local Refinement with L-BFGS-B
    # DE finds a good basin, L-BFGS-B finds the precise peak.
    # We try polishing the best result from DE, and also a few random restarts 
    # polished with L-BFGS-B to cover cases where DE might have missed a better basin.
    
    def run_polishing(start_params, maxiter=2000):
        try:
            res = minimize(
                objective, 
                start_params, 
                method='L-BFGS-B', 
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res
        except Exception:
            return None

    # Polish the DE result
    if time.time() < deadline - 10:
        res_polish = run_polishing(best_params)
        if res_polish is not None and res_polish.fun < -best_score:
            best_score = -res_polish.fun
            best_params = res_polish.x

    # Additional random restarts with polishing if time permits
    # DE is expensive, so we do fewer restarts here, relying on DE's global capability.
    rng = np.random.default_rng(42)
    
    # Define a few structured starts for L-BFGS-B as backup/complement
    starts = [
        np.zeros(100), # Uniform
        best_params.copy(), # Best found so far
    ]
    
    # Add some random starts
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        starts.append(rng.normal(0, 2.0, 100))
        
    # Add a "corner" start (high weight on off-diagonals to maximize x-y spread)
    corner_start = rng.normal(0, 0.5, 100)
    # Indices for (0,9) and (9,0) in P[y,x] (row y, col x)
    # y=0, x=9 -> idx 9
    # y=9, x=0 -> idx 90
    corner_start[9] += 4.0
    corner_start[90] += 4.0
    starts.append(corner_start)
    
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        res = run_polishing(start, maxiter=1000)
        if res is not None:
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_params = res.x

    # Convert best parameters back to P
    log_p_flat = best_params - np.max(best_params)
    p_flat = np.exp(log_p_flat)
    p_flat /= np.sum(p_flat)
    
    P = p_flat.reshape((n, n))
    
    # Final validation
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
