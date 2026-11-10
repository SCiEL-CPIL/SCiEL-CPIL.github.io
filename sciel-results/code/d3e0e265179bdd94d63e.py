# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    K = 10
    N_THETA = 70
    N_PHI = 130
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 600.0
    SAFETY_MARGIN = 10.0
    
    # 1. Setup Quadrature
    x, w_theta = np.polynomial.legendre.leggauss(N_THETA)
    phi = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    
    N_POINTS = N_THETA * N_PHI
    
    # 2. Precompute Spherical Harmonics Y_m at quadrature points
    Y = np.zeros((N_POINTS, 2 * K + 1))
    
    # Compute Normalization Constants N_m
    # N_m = sqrt( (2K+1)/(4 pi) * (K-m)! / (K+m)! )
    N_m = np.zeros(K + 1)
    for m in range(K + 1):
        if m == 0:
            ratio = 1.0
        else:
            ratio = 1.0
            # Ratio is 1 / ((K-m+1) * ... * (K+m))
            for k in range(1, 2 * m + 1):
                ratio /= (K - m + k)
        N_m[m] = np.sqrt((2 * K + 1) / (4.0 * np.pi) * ratio)
        
    # Evaluate Associated Legendre Polynomials P_K^m(x)
    P = np.zeros((K + 1, N_THETA))
    for m in range(K + 1):
        P[m] = lpmv(m, K, x)
        
    # Fill Y matrix
    # x_grid shape (N_THETA, 1), phi_grid shape (1, N_PHI)
    x_grid = x[:, np.newaxis]
    phi_grid = phi[np.newaxis, :]
    
    for m in range(K + 1):
        idx_pos = m + K
        idx_neg = -m + K
        
        N_val = N_m[m]
        P_val = P[m, :, np.newaxis] # (N_THETA, 1)
        
        if m == 0:
            Y_m = N_val * P_val # (N_THETA, 1)
            Y[:, idx_pos] = Y_m.flatten()
        else:
            cos_m_phi = np.cos(m * phi_grid) # (1, N_PHI)
            sin_m_phi = np.sin(m * phi_grid) # (1, N_PHI)
            
            # Y_pos = sqrt(2) * N_m * P_m * cos(m phi)
            Y_pos = np.sqrt(2.0) * N_val * P_val * cos_m_phi # (N_THETA, N_PHI)
            # Y_neg = sqrt(2) * N_m * P_m * sin(m phi)
            Y_neg = np.sqrt(2.0) * N_val * P_val * sin_m_phi # (N_THETA, N_PHI)
            
            Y[:, idx_pos] = Y_pos.flatten()
            Y[:, idx_neg] = Y_neg.flatten()
            
    # Weights for the double sum
    w_phi = 2 * np.pi / N_PHI
    weights = np.kron(w_theta, np.ones(N_PHI) * w_phi) # (N_POINTS,)
    
    # 3. Define Objective and Gradient
    def objective_and_grad(c):
        f = Y.dot(c) # (N_POINTS,)
        
        f2 = f * f
        f4 = f2 * f2
        
        A = np.dot(weights, f4)
        B = np.dot(weights, f2)
        
        # Avoid division by zero
        if B < 1e-12:
            return 1e10, np.zeros_like(c)
            
        val = A / (B * B)
        
        # Gradient
        # dA/dc = 4 * Y^T (w * f^3)
        # dB/dc = 2 * Y^T (w * f)
        
        w_f3 = weights * (f * f2)
        w_f = weights * f
        
        dA = 4.0 * Y.T.dot(w_f3)
        dB = 2.0 * Y.T.dot(w_f)
        
        grad = dA / (B * B) - 2.0 * A * dB / (B ** 3)
        
        return val, grad

    # 4. Optimization
    best_c = np.ones(2 * K + 1, dtype=np.float64)
    best_val = objective_and_grad(best_c)[0]
    
    # Try multiple random starts
    rng = np.random.default_rng(42)
    
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
            
        # Random initialization
        c0 = rng.normal(size=2 * K + 1)
        if np.allclose(c0, 0):
            continue
            
        try:
            res = minimize(
                lambda c: objective_and_grad(c)[0],
                c0,
                jac=lambda c: objective_and_grad(c)[1],
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            
            c_opt = res.x
            val_opt = objective_and_grad(c_opt)[0]
            
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt
        except Exception:
            continue
            
    # Final check
    if not np.all(np.isfinite(best_c)):
        return np.ones(2 * K + 1, dtype=np.float64)
        
    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3:
        best_c = best_c * (1.0 / max_abs_c)
    elif max_abs_c > 1e5:
        best_c = best_c * (1e5 / max_abs_c)
        
    return best_c
# EVOLVE-BLOCK-END
