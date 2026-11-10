import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize
from math import lgamma, log, pi

# Constants
K = 10
M_VALUES = np.arange(-K, K + 1)

# Precompute normalization constants N_m for real orthonormal spherical harmonics
N = np.zeros(2 * K + 1)
for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    log_n_sq = log(21.0 / (4.0 * pi)) + lgamma(11 - abs_m) - lgamma(11 + abs_m)
    N[i] = np.sqrt(np.exp(log_n_sq))

# Quadrature setup
n_theta = 70
n_phi = 130
x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

x_grid, phi_grid_2d = np.meshgrid(x_gl, phi_grid, indexing='ij')

# Precompute the basis functions Y_m(x, phi) for all m
Y_basis = np.zeros((21, n_theta, n_phi))

for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    p_vals = lpmv(abs_m, 10, x_gl) # shape (70,)
    
    if m == 0:
        Y_basis[i] = N[i] * p_vals[:, np.newaxis]
    elif m > 0:
        cos_term = np.cos(m * phi_grid_2d) # shape (70, 130)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * cos_term
    else:
        sin_term = np.sin(abs_m * phi_grid_2d) # shape (70, 130)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * sin_term

# Precompute weights for integration
W_flat = (w_gl[:, np.newaxis] * (2.0 * np.pi / n_phi)).flatten()
Y_flat = Y_basis.reshape(21, -1)

def objective_and_grad(coeffs):
    """
    Computes the objective function R = ||f||_4 / ||f||_2 and its gradient.
    coeffs: shape (21,)
    Returns: (value, gradient)
    """
    f_flat = coeffs @ Y_flat
    
    norm2_sq = np.sum(W_flat * f_flat**2)
    norm4_4 = np.sum(W_flat * f_flat**4)
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return 1e10, np.zeros(21)
        
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    val = norm4 / norm2
    
    # Analytic gradient derivation:
    # Let I2 = ||f||_2^2, I4 = ||f||_4^4.
    # f = Y c.
    # dI2 = 2 * c^T Y^T W Y c
    # dI4 = 4 * c^T Y^T W Y (f^2) c (where f^2 is element-wise)
    # val = I4^0.25 * I2^-0.5
    # dval = 0.25 * I4^-0.75 * dI4 * I2^-0.5 - 0.5 * I4^0.25 * I2^-1.5 * dI2
    # dval = (0.25 * I4^-0.75 * I2^-0.5) * dI4 - (0.5 * I4^0.25 * I2^-1.5) * dI2
    # dI4 = 4 * Y^T W Y (f^2) c
    # dI2 = 2 * Y^T W Y c
    # dval = (0.25 * I4^-0.75 * I2^-0.5 * 4) * Y^T W Y (f^2) c - (0.5 * I4^0.25 * I2^-1.5 * 2) * Y^T W Y c
    # dval = (I4^-0.75 * I2^-0.5) * Y^T W Y (f^2) c - (I4^0.25 * I2^-1.5) * Y^T W Y c
    
    term1_coeff = (norm4_4 ** -0.75) * (norm2_sq ** -0.5)
    term2_coeff = (norm4_4 ** 0.25) * (norm2_sq ** -1.5)
    
    # grad = term1_coeff * Y^T (W * Y * f^2) - term2_coeff * Y^T (W * Y * f)
    # Note: Y is (21, N). W is (N,). f is (N,).
    # Y^T (W * Y * f^2) is (21,).
    # We can compute this as:
    # temp1 = W * (Y_flat * f_flat**2) -> (21, N)
    # grad1 = Y_flat.T @ temp1 -> (21,)
    # temp2 = W * (Y_flat * f_flat) -> (21, N)
    # grad2 = Y_flat.T @ temp2 -> (21,)
    
    grad1 = Y_flat.T @ (W_flat * (Y_flat * f_flat**2))
    grad2 = Y_flat.T @ (W_flat * (Y_flat * f_flat))
    
    grad = term1_coeff * grad1 - term2_coeff * grad2
    
    return val, grad

def run_search(time_budget):
    start_time = time.time()
    best_coeffs = None
    best_val = np.inf
    
    # Helper to check time
    def time_left():
        return time_budget - (time.time() - start_time)
    
    # Initial candidates
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(21))
    
    # 2. Random Gaussian
    for _ in range(20):
        candidates.append(np.random.randn(21))
        
    # 3. Structured patterns
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) <= 2, 1, 0))
    candidates.append(np.ones(21) * np.where(M_VALUES % 2 == 0, 1, 0))
    candidates.append(np.ones(21) * np.where(M_VALUES % 2 != 0, 1, 0))
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) >= 8, 1, 0))
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) <= 5, 1, 0))
    candidates.append(np.ones(21) * np.where(np.abs(M_VALUES) >= 5, 1, 0))
    
    # 4. Alternating signs patterns
    for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
        c = np.ones(21)
        for i, m in enumerate(M_VALUES):
            c[i] = np.cos(m * phase)
        candidates.append(c)
        
    # 5. Add some more diverse random starts with different scales
    for scale in [0.5, 1.0, 2.0, 5.0]:
        candidates.append(np.random.randn(21) * scale)
    
    # Evaluate initial candidates and select the best few for local optimization
    # Sort by objective value
    scored_candidates = []
    for c in candidates:
        val, _ = objective_and_grad(c)
        scored_candidates.append((val, c))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 10 for local refinement
    top_candidates = [c for _, c in scored_candidates[:10]]
    
    # Local optimization using L-BFGS-B
    for c0 in top_candidates:
        if time_left() < 10.0:
            break
            
        try:
            result = minimize(
                objective_and_grad, 
                c0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            if result.success:
                val_res, grad_res = objective_and_grad(result.x)
                if val_res < best_val:
                    best_val = val_res
                    best_coeffs = result.x.copy()
        except Exception:
            continue
            
    # If we still have time, do a second round of refinement from the best point
    if best_coeffs is not None and time_left() > 30.0:
        try:
            # Perturb slightly and re-optimize to ensure we are at a true local min
            # or use a tighter tolerance
            c_start = best_coeffs + np.random.randn(21) * 0.1
            result = minimize(
                objective_and_grad, 
                c_start, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
            )
            
            if result.success:
                val_res, _ = objective_and_grad(result.x)
                if val_res < best_val:
                    best_val = val_res
                    best_coeffs = result.x.copy()
        except Exception:
            pass
            
    return best_coeffs

def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    # Time budget management
    start_time = time.time()
    max_time = 580.0 # Safety margin from 630s (leave 50s buffer)
    
    # Run search
    try:
        best_coeffs = run_search(max_time)
    except Exception as e:
        # Fallback to baseline if something goes wrong
        best_coeffs = np.ones(21)
        
    # Final validation and cleanup
    if best_coeffs is None or not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
        
    # Ensure size is 21
    if len(best_coeffs) != 21:
        best_coeffs = np.ones(21)
        
    # Check max abs coefficient constraints
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        # Normalize to be within range, e.g., scale to max 1.0
        if max_c > 0:
            best_coeffs = best_coeffs / max_c * 1.0
        else:
            best_coeffs = np.ones(21)
            
    return best_coeffs
