"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Represent the step function as a linear combination of even-order Chebyshev polynomials.
Use a two-phase approach: first use penalty method to get a good starting point, then refine
with logarithmic barrier method for smooth non-negativity constraint handling.
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

def chebyshev_basis(x, max_order):
    """
    Generate Chebyshev polynomial basis values for orders 0 to max_order.
    x should be in [-1/4, 1/4]. Returns array of shape (len(x), max_order+1)
    """
    x_scaled = x / 0.25  # Map to [-1, 1]
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    
    n_terms = max_order + 1
    basis = np.zeros((len(x), n_terms))
    
    # T_0(x) = 1
    basis[:, 0] = 1.0
    # T_1(x) = x
    if n_terms > 1:
        basis[:, 1] = x_scaled
        
    for k in range(2, n_terms):
        basis[:, k] = 2 * x_scaled * basis[:, k-1] - basis[:, k-2]
        
    return basis

def objective_barrier(coeffs, max_order, x_grid, basis, mu):
    """
    Objective function with logarithmic barrier for non-negativity.
    f = basis @ coeffs must be > 0 (barrier pushes away from 0).
    """
    f = basis @ coeffs
    
    # Check if any values are too close to zero or negative
    min_val = np.min(f)
    if min_val <= 0:
        # Large penalty if we violate the barrier region
        return 1e10 + (min_val**2)
    
    # Logarithmic barrier: -mu * sum(log(f))
    barrier_term = -mu * np.sum(np.log(f))
    
    # Compute the actual score
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    
    conv = np.convolve(f, f, mode='full')
    score = np.max(conv) / (s * s)
    
    return score + barrier_term

def objective_penalty(coeffs, max_order, x_grid, basis, penalty_weight):
    """
    Objective function with quadratic penalty for non-negativity violations.
    """
    f = basis @ coeffs
    min_val = np.min(f)
    
    # Penalty for non-negativity
    penalty = 0.0
    if min_val < 0:
        neg_parts = f[f < 0]
        penalty = penalty_weight * np.sum(neg_parts**2)
    
    # Compute score using clipped f
    f_clipped = np.maximum(f, 0.0)
    s = np.sum(f_clipped)
    if s < 1e-15:
        return 1e10 + penalty
    
    conv = np.convolve(f_clipped, f_clipped, mode='full')
    score = np.max(conv) / (s * s)
    
    return score + penalty

def construct():
    start_time = time.time()
    
    x_grid = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different max orders
    # Even orders are often sufficient for symmetric functions, but let's allow all orders
    # to capture any asymmetry if it helps (though the problem is symmetric, optimal is likely symmetric).
    # We will enforce symmetry by only using even orders to reduce parameters and respect symmetry.
    # If we use only even orders, f(x) = f(-x).
    
    for max_order in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        # Use only even orders: 0, 2, 4, ..., max_order
        # This ensures f is even.
        even_orders = list(range(0, max_order + 1, 2))
        n_params = len(even_orders)
        
        # Build basis for these specific orders
        full_basis = chebyshev_basis(x_grid, max_order)
        basis = full_basis[:, even_orders]
        
        best_params = None
        best_obj = np.inf
        
        # Phase 1: Penalty method to get a good starting point
        n_restarts = 15
        for restart in range(n_restarts):
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            
            rng = np.random.RandomState(42 + max_order * 100 + restart)
            
            # Initialization
            if restart == 0:
                x0 = np.zeros(n_params)
                x0[0] = 1.0
            else:
                x0 = rng.uniform(-0.3, 0.3, n_params)
                x0[0] += 1.0
            
            # Optimize using Powell with penalty
            try:
                result = minimize(
                    objective_penalty, x0, args=(max_order, x_grid, basis, 500.0),
                    method='Powell',
                    options={'maxiter': 300, 'ftol': 1e-9, 'xtol': 1e-9}
                )
                
                f_test = basis @ result.x
                if np.min(f_test) < -1e-6:
                    f_test = np.maximum(f_test, 0.0)
                
                actual_score = compute_score(f_test)
                
                if actual_score < best_obj:
                    best_obj = actual_score
                    best_params = result.x.copy()
                    
            except Exception:
                pass
        
        # Phase 2: Barrier method refinement from best penalty solution
        if best_params is not None:
            for mu in [0.1, 0.01, 0.001, 0.0001]:
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                
                # Start from the best penalty solution
                x0 = best_params.copy()
                
                try:
                    result = minimize(
                        objective_barrier, x0, args=(max_order, x_grid, basis, mu),
                        method='Powell',
                        options={'maxiter': 200, 'ftol': 1e-10, 'xtol': 1e-10}
                    )
                    
                    f_test = basis @ result.x
                    if np.min(f_test) < -1e-8:
                        f_test = np.maximum(f_test, 0.0)
                    
                    actual_score = compute_score(f_test)
                    
                    if actual_score < best_obj:
                        best_obj = actual_score
                        best_params = result.x.copy()
                        
                except Exception:
                    pass
        
        if best_params is not None:
            f = basis @ best_params
            f = np.maximum(f, 0.0)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement on the grid
    if time.time() - start_time < TIME_LIMIT - 15:
        current_h = best_f.copy()
        
        # Normalize to have mean 1
        s = np.sum(current_h)
        if s > 1e-12:
            current_h = current_h * (N / s)
        
        # Multi-scale local search
        for outer_iter in range(150):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            improved = False
            
            # Try different step sizes
            for step_scale in [0.05, 0.1, 0.2, 0.5]:
                # Pick random indices to perturb
                n_perturb = max(1, N // 50)
                indices = np.random.choice(N, size=n_perturb, replace=False)
                
                for j in indices:
                    old_val = current_h[j]
                    
                    # Try scaling up and down
                    for scale in [1 - step_scale, 1 + step_scale]:
                        h_new = current_h.copy()
                        h_new[j] = old_val * scale
                        h_new = np.maximum(h_new, 0.0)
                        
                        s_new = np.sum(h_new)
                        if s_new < 1e-12:
                            continue
                        h_new = h_new * (N / s_new)
                        
                        conv_new = np.convolve(h_new, h_new, mode='full')
                        val_new = np.max(conv_new) / (N * N)
                        
                        if val_new < best_score - 1e-12:
                            current_h = h_new
                            best_score = val_new
                            best_f = current_h.copy()
                            improved = True
                            break
                    if improved:
                        break
                if improved:
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
