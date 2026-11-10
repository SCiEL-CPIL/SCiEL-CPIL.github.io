# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import dual_annealing

def _precompute_basis():
    k = 10
    N_theta = 70
    N_phi = 130
    
    theta_grid, w_theta = np.polynomial.legendre.leggauss(N_theta)
    phi_grid = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    Y_matrix = np.zeros((N_theta, N_phi, 2 * k + 1), dtype=np.float64)
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        P = lpmv(abs_m, k, theta_grid)
        
        if abs_m == 0:
            fact_ratio = 1.0
        else:
            prod = 1.0
            for j in range(10 - abs_m + 1, 10 + abs_m + 1):
                prod *= j
            fact_ratio = 1.0 / prod
        
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * fact_ratio)
        
        if m == 0:
            Y_vals = N_m * P
            Y_matrix[:, :, m_idx] = Y_vals[:, np.newaxis] * np.ones((1, N_phi))[np.newaxis, :]
        elif m > 0:
            cos_term = np.cos(m * phi_grid)
            Y_vals = np.sqrt(2.0) * N_m * P
            Y_matrix[:, :, m_idx] = Y_vals[:, np.newaxis] * cos_term[np.newaxis, :]
        else:
            sin_term = np.sin(abs_m * phi_grid)
            Y_vals = np.sqrt(2.0) * N_m * P
            Y_matrix[:, :, m_idx] = Y_vals[:, np.newaxis] * sin_term[np.newaxis, :]
            
    # Flatten for fast matrix-vector multiplication
    Y_flat = Y_matrix.reshape(N_theta * N_phi, 2 * k + 1)
    
    # Precompute weights
    # integral = sum_i w_i * (2*pi/N_phi) * sum_j f_ij
    # = sum_{ij} w_i * (2*pi/N_phi) * f_ij
    W_flat = np.repeat(w_theta, N_phi) * (2.0 * np.pi / N_phi)
    
    return W_flat, Y_flat

def _compute_R(c, W_flat, Y_flat):
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-15:
        return 1.0
    c_norm = c / norm_c
    
    f = Y_flat @ c_norm
    
    f2 = f ** 2
    f4 = f ** 4
    
    norm2_sq = np.dot(W_flat, f2)
    norm4_4 = np.dot(W_flat, f4)
    
    if norm2_sq <= 0:
        return 1.0
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2

def rudin_problem10() -> np.ndarray:
    W_flat, Y_flat = _precompute_basis()
    
    def objective(c):
        return _compute_R(c, W_flat, Y_flat)
    
    x0 = np.ones(21)
    bounds = [(-1.0, 1.0) for _ in range(21)]
    
    result = dual_annealing(
        objective, 
        x0, 
        bounds=bounds, 
        maxiter=5000, 
        restarts=5,
        seed=42
    )
    
    c_opt = result.x
    
    max_abs = np.max(np.abs(c_opt))
    if max_abs < 1e-15:
        c_opt = np.ones(21)
    else:
        c_opt = c_opt / max_abs
        
    return c_opt

# EVOLVE-BLOCK-END
