"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Represent the step function as the square of a Chebyshev polynomial series.
f(x) = (sum a_k T_k(x))^2 ensures f(x) >= 0.
Optimize the coefficients a_k using a gradient-based method (L-BFGS-B) to minimize
the normalized max autoconvolution. This enforces non-negativity via the parameterization
(barrier-like effect) and allows for faster convergence than derivative-free methods
on the original space.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def chebyshev_basis_even(x, max_order):
    """
    Generate even Chebyshev polynomial basis values T_0, T_2, ..., T_{2*max_order}.
    x should be in [-1/4, 1/4].
    """
    x_scaled = x / 0.25
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    
    max_k = 2 * max_order
    T = np.zeros((len(x), max_k + 1))
    T[:, 0] = 1.0
    if max_k >= 1:
        T[:, 1] = x_scaled
    for k in range(2, max_k + 1):
        T[:, k] = 2 * x_scaled * T[:, k-1] - T[:, k-2]
        
    even_indices = np.arange(0, max_k + 1, 2)
    return T[:, even_indices]

def objective_barrier(coeffs, basis, mu):
    """
    Objective function with barrier for non-negativity.
    f = basis @ coeffs
    Normalizes f to sum 1, computes score, adds barrier term.
    """
    f = basis @ coeffs
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    
    f_norm = f / s
    
    # Check for negative values
    if np.any(f_norm < -1e-12):
        return 1e10
    
    # Penalty for values close to zero (barrier)
    eps = 1e-12
    barrier = -mu * np.sum(np.log(f_norm + eps))
    
    main_score = compute_score(f_norm)
    
    return main_score + barrier

def construct():
    start_time = time.time()
    
    x_grid = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Orders to try: max_order corresponds to highest T_{2*max_order}
    orders = [2, 4, 6, 8, 10, 12, 15, 20]
    
    for max_order in orders:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        basis = chebyshev_basis_even(x_grid, max_order)
        n_params = basis.shape[1]
        
        best_params = None
        best_obj = np.inf
        
        # Try different barrier strengths
        mus = [1e-4, 1e-3, 1e-2]
        
        for mu in mus:
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            # Initial guess: constant
            x0 = np.zeros(n_params)
            x0[0] = 1.0
            
            # Also try a small random perturbation
            rng = np.random.RandomState(42 + max_order)
            x0_rand = x0.copy()
            x0_rand[1:] += rng.randn(n_params - 1) * 0.1
            
            for x_start in [x0, x0_rand]:
                try:
                    result = minimize(
                        objective_barrier, x_start, args=(basis, mu),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    
                    f_test = basis @ result.x
                    s_test = np.sum(f_test)
                    if s_test > 1e-15:
                        f_norm_test = f_test / s_test
                        if np.all(f_norm_test >= -1e-12):
                            actual_score = compute_score(f_norm_test)
                            if actual_score < best_obj:
                                best_obj = actual_score
                                best_params = result.x.copy()
                except Exception:
                    pass
        
        if best_params is not None:
            f = basis @ best_params
            s = np.sum(f)
            if s > 1e-15:
                f = f / s
                actual_score = compute_score(f)
                if actual_score < best_score:
                    best_f = f
                    best_score = actual_score

    # Local refinement
    if time.time() - start_time < TIME_LIMIT - 20:
        current_h = best_f.copy()
        s = np.sum(current_h)
        if s > 1e-12:
            current_h = current_h * (N / s)
        
        for outer_iter in range(100):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            improved = False
            for _ in range(10):
                j = np.random.randint(0, N)
                old_val = current_h[j]
                
                for scale in [0.9, 1.1, 0.8, 1.2]:
                    h_new = current_h.copy()
                    h_new[j] = old_val * scale
                    h_new = np.maximum(h_new, 0.0)
                    
                    s_new = np.sum(h_new)
                    if s_new < 1e-12:
                        continue
                    h_new = h_new * (N / s_new)
                    
                    conv_new = np.convolve(h_new, h_new, mode='full')
                    val_new = np.max(conv_new) / (N * N)
                    
                    if val_new < best_score:
                        current_h = h_new
                        best_score = val_new
                        best_f = current_h.copy()
                        improved = True
                        break
            if not improved:
                break
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
