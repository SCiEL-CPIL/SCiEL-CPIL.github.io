import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c for degree-10 spherical harmonics minimizing ||f||_4 / ||f||_2.
    Uses Nelder-Mead simplex optimization with multiple restarts.
    """
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # --- 1. Precompute Spherical Harmonics ---
    # 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    x_grid, phi_grid = np.meshgrid(x_gl, phi_vals, indexing='ij')
    x_flat = x_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Effective weights for integration over sphere
    # Integral = sum_i w_i * (2*pi/n_phi) * sum_j f(x_i, phi_j)
    W_eff = np.repeat(w_gl, n_phi) * (2.0 * np.pi / n_phi)
    
    # Compute normalization constants N_m
    N = np.zeros(n_coeffs)
    for m in range(0, k + 1):
        norm_factor = (2 * k + 1) / (4.0 * np.pi) * factorial(k - m) / factorial(k + m)
        N[m] = np.sqrt(norm_factor)
        if m > 0:
            N[-m] = N[m]
            
    # Build Y matrix (n_points, n_coeffs)
    n_points = len(x_flat)
    Y_matrix = np.zeros((n_points, n_coeffs))
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        if m == 0:
            P_val = lpmv(0, k, x_flat)
            Y_matrix[:, m_idx] = N[0] * P_val
        elif m > 0:
            P_val = lpmv(m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[m] * P_val * np.cos(m * phi_flat)
        else:
            abs_m = -m
            P_val = lpmv(abs_m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[abs_m] * P_val * np.sin(abs_m * phi_flat)
            
    # --- 2. Define Objective Function ---
    def objective(coeffs):
        # coeffs is shape (21,)
        f = Y_matrix @ coeffs
        f2 = f * f
        f4 = f2 * f2
        
        l2_sq = np.dot(W_eff, f2)
        if l2_sq < 1e-12:
            return 1e10
            
        l4 = np.dot(W_eff, f4)
        
        # Minimize R^4 = L4 / L2^2
        # Note: R = (L4)^(1/4) / (L2_sq)^(1/2)
        # R^4 = L4 / L2_sq^2
        return l4 / (l2_sq * l2_sq)
        
    # --- 3. Multi-Start Nelder-Mead Optimization ---
    best_score = np.inf
    best_coeffs = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Single modes
    for m in range(-k, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        inits.append(init)
        
    # 2. Symmetric pairs (cos only, sin only, mixed)
    for m in range(1, k + 1):
        # Cosine mode
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        inits.append(init)
        
        # Sine mode
        init = np.zeros(n_coeffs)
        init[-m + k] = 1.0
        inits.append(init)
        
        # Mixed
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        init[-m + k] = 1.0
        inits.append(init)
        
    # 3. Random vectors (dense)
    for _ in range(50):
        init = np.random.randn(n_coeffs)
        init /= np.linalg.norm(init)
        inits.append(init)
        
    # 4. Sparse random
    for _ in range(50):
        init = np.random.randn(n_coeffs)
        mask = np.random.rand(n_coeffs) < 0.3
        init[~mask] = 0
        if np.linalg.norm(init) > 1e-6:
            init /= np.linalg.norm(init)
        inits.append(init)
        
    # 5. "Flat" initializations: try to make f constant-like
    # (Though degree 10 cannot be constant, we try to minimize high freq components)
    # This is just a heuristic, random is usually better.
    
    # Shuffle
    np.random.shuffle(inits)
    
    # Optimization Loop
    # Nelder-Mead parameters
    nm_options = {
        'maxiter': 500,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True
    }
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Check if we have enough time for a few iterations
        # Each iteration is ~43 evals. 500 iters ~ 21500 evals.
        # Eval time ~ 100us? 21500 * 100us = 2s.
        # So we can do many restarts.
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options=nm_options)
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
                
        except Exception:
            continue
            
    # --- 4. Refine Best Solution ---
    # Perform a final high-precision Nelder-Mead run on the best solution found
    if time.time() < deadline - 10:
        try:
            nm_options_refine = {
                'maxiter': 2000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
            res = minimize(objective, best_coeffs, method='Nelder-Mead', options=nm_options_refine)
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # --- 5. Validate and Return ---
    # Ensure validity constraints
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(n_coeffs)
        
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3:
        best_coeffs *= (1.0 / max_c)
    elif max_c > 1e5:
        best_coeffs /= max_c
        
    return best_coeffs
