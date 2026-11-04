"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_genlaguerre
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 580  # safety margin before 630s kill
    
    m = 12
    n = 1
    max_degree = 2 * m  # 24
    
    def compute_coeffs(roots):
        """Solve for coefficients of Laguerre combination with double roots."""
        roots = np.sort(roots)
        if np.any(roots <= 0):
            return None
        
        A = np.zeros((24, 24))
        b = np.zeros(24)
        
        for i, r in enumerate(roots):
            for k in range(1, max_degree + 1):
                A[2*i][k-1] = eval_genlaguerre(k, n, r)
            b[2*i] = -1.0
            
            for k in range(1, max_degree + 1):
                A[2*i+1][k-1] = -eval_genlaguerre(k-1, n+1, r)
            b[2*i+1] = 0.0
        
        try:
            coeffs_1_24 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        
        coeffs = np.zeros(max_degree + 1)
        coeffs[0] = 1.0
        coeffs[1:] = coeffs_1_24
        return coeffs
    
    def find_last_sign_change(coeffs, roots):
        """Find the last sign change of P(x) = sum coeffs[k] L_k^n(x)."""
        x_max = max(roots) * 10 + 100
        # Use adaptive sampling
        x_samples = np.linspace(0, x_max, 200000)
        
        P_vals = np.zeros_like(x_samples)
        for k in range(max_degree + 1):
            P_vals += coeffs[k] * eval_genlaguerre(k, n, x_samples)
        
        signs = np.sign(P_vals)
        nonzero_mask = signs != 0
        signs_nz = signs[nonzero_mask]
        x_nz = x_samples[nonzero_mask]
        
        if len(signs_nz) < 2:
            return None
        
        sign_changes = np.where(np.diff(signs_nz) != 0)[0]
        if len(sign_changes) == 0:
            return None
        
        last_sc_idx = sign_changes[-1]
        x_lo = x_nz[last_sc_idx]
        x_hi = x_nz[last_sc_idx + 1]
        
        # Bisection
        P_lo = P_vals[np.searchsorted(x_samples, x_lo)]
        for _ in range(60):
            x_mid = (x_lo + x_hi) / 2
            P_mid = 0.0
            for k in range(max_degree + 1):
                P_mid += coeffs[k] * eval_genlaguerre(k, n, x_mid)
            if P_mid * P_lo > 0:
                x_lo = x_mid
                P_lo = P_mid
            else:
                x_hi = x_mid
        
        return (x_lo + x_hi) / 2
    
    def objective(roots):
        roots = np.sort(roots)
        if np.any(roots <= 0):
            return 1e10
        coeffs = compute_coeffs(roots)
        if coeffs is None:
            return 1e10
        lsc = find_last_sign_change(coeffs, roots)
        if lsc is None:
            return 1e10
        return lsc
    
    # Multiple restarts
    best_score = 1e10
    best_roots = None
    
    rng = np.random.RandomState(42)
    
    initial_guesses = [
        np.linspace(1, 20, 12),
        np.linspace(0.5, 15, 12),
        4.0 * np.arange(1, 13),
        np.sqrt(np.linspace(1, 100, 12)),
        np.logspace(0, 1.5, 12),
        np.linspace(2, 25, 12),
        rng.uniform(0.5, 20, 12),
        rng.uniform(1, 30, 12),
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, guess, method='L-BFGS-B', 
                             bounds=[(0.01, None)] * 12,
                             options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    # Refinement
    if best_roots is not None and time.time() < deadline:
        try:
            result = minimize(objective, best_roots, method='L-BFGS-B',
                             bounds=[(0.01, None)] * 12,
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    if best_roots is not None and time.time() < deadline:
        try:
            result = minimize(objective, best_roots, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    if best_roots is None:
        return [4.0 * (i + 1) for i in range(12)]
    
    return [float(r) for r in np.sort(best_roots)]
# EVOLVE-BLOCK-END
