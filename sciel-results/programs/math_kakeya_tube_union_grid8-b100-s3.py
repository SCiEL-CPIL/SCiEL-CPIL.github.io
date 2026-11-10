# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).
    
    Strategy:
    1. Recognize that the union volume V = integral_0^1 Ux(c) * Uy(c) dc.
    2. The current program minimizes E[Ux] * E[Uy] by optimizing independently, 
       which is suboptimal because Cov(Ux, Uy) is ignored.
    3. Exploit symmetry: assume x_i = y_i. Then V = integral_0^1 Ux(c)^2 dc.
    4. Optimize the 1D positions x to minimize the integral of the SQUARE of 
       the 1D union length.
    5. Use a robust optimization approach (Nelder-Mead with careful initialization 
       and high-resolution MC for the objective).
    """
    N = 8
    deadline = time.time() + 580.0  # Safety margin
    
    # 1. Analytical Initialization
    # Centered shear alignment is a good start.
    k = np.arange(N, dtype=np.float64)
    x_init = -k / (2.0 * N)
    
    # 2. Monte Carlo Setup for Objective Function
    # Objective: f(x) = integral_0^1 Ux(c)^2 dc
    # We estimate this using MC samples of c.
    
    M = 200_000  # Sample size for objective estimation
    rng = np.random.default_rng(12345)
    c_samples = rng.uniform(0, 1, M)
    
    # Precompute shear coefficients
    i_vals = np.arange(N, dtype=np.float64)
    shear_coeffs = i_vals / N  # shape (N,)
    
    def compute_1d_union_length_sq(x, c_samples):
        """
        Computes the average of the SQUARE of the union length for given x and c_samples.
        x: shape (N,)
        c_samples: shape (M,)
        Returns: scalar (average of U(c)^2)
        """
        # L: (N, M)
        L = x[:, None] + c_samples[None, :] * shear_coeffs[:, None]
        R = L + 1.0 / N
        
        # Transpose to (M, N)
        L_t = L.T
        R_t = R.T
        
        # Sort intervals by start point L
        order = np.argsort(L_t, axis=1)
        L_sorted = np.take_along_axis(L_t, order, axis=1)
        R_sorted = np.take_along_axis(R_t, order, axis=1)
        
        # Compute union length for each c using vectorized merge
        # Union Length = Sum_{k=0}^{N-1} max(0, R_k - max(L_k, max_{j<k} R_j))
        
        max_r_up_to = np.maximum.accumulate(R_sorted, axis=1)
        
        # Create array of max R before current index
        max_r_before = np.empty_like(R_sorted)
        max_r_before[:, 0] = -np.inf
        max_r_before[:, 1:] = max_r_up_to[:, :-1]
        
        effective_start = np.maximum(L_sorted, max_r_before)
        contributions = np.maximum(0, R_sorted - effective_start)
        
        union_lens = np.sum(contributions, axis=1)
        
        # We want the mean of the SQUARE of the union lengths
        return np.mean(union_lens ** 2)

    def objective(x):
        if time.time() > deadline:
            return 1e9
        return compute_1d_union_length_sq(x, c_samples)

    # 3. Optimization
    # We use Nelder-Mead but with a larger initial simplex and more iterations 
    # since the landscape might be smoother for the squared objective.
    
    x0 = x_init.copy()
    
    # Try a few different initializations to avoid local minima
    # 1. Centered
    # 2. Slightly shifted
    # 3. "Compressed" fan
    
    candidates = [
        x_init.copy(),
        x_init + 0.01 * np.random.normal(size=N),
        -k / (2.0 * N) * 0.5, # Compressed
        -k / (2.0 * N) * 1.5  # Expanded
    ]
    
    best_x = x0
    best_val = np.inf
    
    for x_start in candidates:
        if time.time() > deadline - 10:
            break
            
        res = minimize(objective, x_start, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_val and np.all(np.isfinite(res.x)):
            best_val = res.fun
            best_x = res.x
    
    # 4. Refinement with higher accuracy MC
    # Use a larger sample size for the final evaluation to ensure the result is robust
    M_final = 500_000
    c_final = rng.uniform(0, 1, M_final)
    
    def objective_final(x):
        return compute_1d_union_length_sq(x, c_final)
        
    if time.time() < deadline - 5:
        res_final = minimize(objective_final, best_x, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12})
        if res_final.fun < best_val:
            best_x = res_final.x
            best_val = res_final.fun

    # 5. Construct the 64x2 array
    # Row k is tube (i, j) with i = k // 8 and j = k % 8.
    # Base position is (x_i, x_j) since we assumed x=y.
    
    positions = np.zeros((64, 2), dtype=np.float64)
    for k in range(64):
        i = k // 8
        j = k % 8
        positions[k, 0] = best_x[i]
        positions[k, 1] = best_x[j]
        
    return positions

# EVOLVE-BLOCK-END
