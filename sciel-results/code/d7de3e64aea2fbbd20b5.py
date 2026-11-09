# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Parameterize the distribution P[x,y] using a quadratic log-likelihood:
       log P[x,y] = c0 + c1*x + c2*y + c3*x^2 + c4*y^2 + c5*xy
       This reduces the search space from 100 dimensions to 6.
    2. Use Differential Evolution for global optimization over these 6 parameters.
    3. This avoids local minima traps common in gradient-based methods on the full simplex.
    """
    n = 10
    deadline = time.time() + 590.0  # Safety margin
    
    # Grid coordinates
    # P[y, x] -> y is row (0..9), x is col (0..9)
    ys, xs = np.mgrid[0:n, 0:n]
    xs_flat = xs.flatten()
    ys_flat = ys.flatten()
    
    # Linear forms for entropy calculations
    d_val = (xs_flat - ys_flat)
    m0_val = xs_flat
    m1_val = (xs_flat + ys_flat)
    m2_val = (xs_flat + 2 * ys_flat)
    
    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    # Precompute feature matrix for the parametric model
    # Features: [1, x, y, x^2, y^2, xy]
    features = np.column_stack([
        np.ones(n*n),
        xs_flat,
        ys_flat,
        xs_flat**2,
        ys_flat**2,
        xs_flat * ys_flat
    ])
    
    def compute_entropies_from_p(p_flat):
        """
        p_flat: shape (100,) normalized probability vector
        Returns: D, M
        """
        p_d = np.bincount(d_inv, weights=p_flat, minlength=len(d_vals))
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=len(m0_vals))
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=len(m1_vals))
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=len(m2_vals))
        
        def shannon(p_dist):
            p_nonzero = p_dist[p_dist > 1e-18]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m_m2) if False else shannon(p_m2) # Typo check: p_m2
        
        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def objective(params):
        """
        params: shape (6,) coefficients [c0, c1, c2, c3, c4, c5]
        Returns: -D/M (to be minimized)
        """
        # Calculate log-probabilities
        log_p = features @ params
        
        # Normalize to get probabilities
        # Subtract max for numerical stability
        log_p_shifted = log_p - np.max(log_p)
        p = np.exp(log_p_shifted)
        sum_p = np.sum(p)
        
        if sum_p < 1e-18:
            return 1e9
            
        p = p / sum_p
        
        # Check if we have enough time
        if time.time() > deadline:
            return 1e9
            
        H_d, M = compute_entropies_from_p(p)
        
        if M < 1e-9:
            return 1e9
            
        return -H_d / M

    # Define bounds for the 6 parameters
    # The coefficients can be negative or positive.
    # A reasonable range is [-5, 5] to allow for significant shape changes without overflow issues.
    bounds = [(-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5)]
    
    # Run Differential Evolution
    # maxiter and popsize control the search effort.
    # 100 iterations with popsize 15 is ~1500 function evaluations.
    # Each evaluation is fast (vectorized numpy ops).
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-6,
            polish=True  # Use L-BFGS-B to polish the result
        )
        
        if result.success and time.time() < deadline:
            best_params = result.x
        else:
            # Fallback to uniform if DE fails or times out
            best_params = np.zeros(6)
            
    except Exception:
        best_params = np.zeros(6)
    
    # Evaluate the best parameters found
    log_p_best = features @ best_params
    log_p_best_shifted = log_p_best - np.max(log_p_best)
    p_best = np.exp(log_p_best_shifted)
    p_best = p_best / np.sum(p_best)
    
    # Reshape to 10x10
    P = p_best.reshape((n, n))
    
    # Ensure valid
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
