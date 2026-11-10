import numpy as np
from scipy.special import lpmv, factorial
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
    
    # Compute N_m
    N_m = np.zeros(K + 1)
    for m in range(K + 1):
        num = (2 * K + 1) / (4 * np.pi) * factorial(K - m) / factorial(K + m)
        N_m[m] = np.sqrt(num)
        
    # Evaluate P_10^m(x)
    P = np.zeros((K + 1, N_THETA))
    for m in range(K + 1):
        P[m] = lpmv(m, K, x)
        
    # Fill Y matrix
    x_grid = x[:, np.newaxis]
    phi_grid = phi[np.newaxis, :]
    
    for m in range(K + 1):
        idx_pos = m + K
        idx_neg = -m + K
        
        N_val = N_m[m]
        P_val = P[m, :, np.newaxis] # (N_THETA, 1)
        
        if m == 0:
            Y_m = N_val * P_val * np.ones((1, N_PHI))
            Y[:, idx_pos] = Y_m.flatten()
        else:
            cos_m_phi = np.cos(m * phi_grid) # (1, N_PHI)
            sin_m_phi = np.sin(m * phi_grid) # (1, N_PHI)
            
            Y_pos = np.sqrt(2.0) * N_val * P_val * cos_m_phi # (N_THETA, N_PHI)
            Y_neg = np.sqrt(2.0) * N_val * P_val * sin_m_phi # (N_THETA, N_PHI)
            
            Y[:, idx_pos] = Y_pos.flatten()
            Y[:, idx_neg] = Y_neg.flatten()
            
    # Weights for the double sum
    w_phi = 2 * np.pi / N_PHI
    weights = np.kron(w_theta, np.ones(N_PHI) * w_phi) # (N_POINTS,)
    
    # 3. Define Objective and Gradient for Constrained Problem
    # Minimize J(c) = int f^4 dOmega subject to sum(c_i^2) = 1
    # Since Y are orthonormal, int f^2 = sum c_i^2 = 1.
    # So R^2 = J(c) / 1^2 = J(c).
    # Minimizing R is equivalent to minimizing J(c).
    
    def objective(c):
        f = Y.dot(c)
        return np.dot(weights, f**4)
        
    def gradient(c):
        f = Y.dot(c)
        # d/dc int f^4 = 4 int f^3 Y_m = 4 Y^T (w * f^3)
        return 4.0 * Y.T.dot(weights * (f**3))
        
    def constraint_fun(c):
        return np.dot(c, c) - 1.0
        
    def constraint_jac(c):
        return 2.0 * c

    # 4. Optimization
    # We use SLSQP which supports equality constraints.
    # We try multiple random starts to find the global minimum.
    
    best_c = np.ones(2 * K + 1) / np.sqrt(2 * K + 1)
    best_val = objective(best_c)
    
    rng = np.random.default_rng(42)
    
    # Number of restarts
    # Each SLSQP run is fast. We can afford many.
    N_RESTARTS = 50
    
    for i in range(N_RESTARTS):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
            
        # Random start on the unit sphere
        c0 = rng.normal(size=2 * K + 1)
        c0 /= np.linalg.norm(c0)
        
        # Also try specific symmetric starts
        if i == 0:
            c0 = np.zeros(2 * K + 1)
            c0[K] = 1.0 # m=0 only
        elif i == 1:
            c0 = np.zeros(2 * K + 1)
            c0[K-1] = 1/np.sqrt(2)
            c0[K+1] = 1/np.sqrt(2) # m=1 cos/sin mix
            
        try:
            res = minimize(
                objective,
                c0,
                jac=gradient,
                method='SLSQP',
                constraints=[{'type': 'eq', 'fun': constraint_fun, 'jac': constraint_jac}],
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            c_opt = res.x
            # Ensure it's on the sphere (SLSQP might drift slightly)
            c_opt /= np.linalg.norm(c_opt)
            
            val_opt = objective(c_opt)
            
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
