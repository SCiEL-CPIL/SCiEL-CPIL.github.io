import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A on quadrature grid.
    2. Define objective R(c) = ||f||_4 / ||f||_2 and its analytic gradient.
    3. Use a multi-start local optimization strategy (L-BFGS-B) with diverse 
       initial guesses to avoid local minima.
    4. Utilize the full time budget to run many optimizations.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

    # 1. Setup Quadrature Grid
    # 70-point Gauss-Legendre in mu = cos(theta)
    mu, w = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    # Flatten grid
    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    
    n_points = len(mu_flat)

    # 2. Construct Basis Functions
    A = np.zeros((n_points, n_m))
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs) / factorial(10 + m_abs))
        return np.sqrt(val)
    
    # m = 0
    m = 0
    idx = 10 
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P
    
    # m > 0
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)
        
    # Precompute weights for norms
    # Objective: Minimize R(c) = ||f||_4 / ||f||_2
    # f = A @ c
    # ||f||_2^2 = sum w_i f_i^2 d_phi
    # ||f||_4^4 = sum w_i f_i^4 d_phi
    
    def compute_norms(c):
        f_vals = A @ c
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        return norm2_sq, norm4_4, f_vals

    def objective_and_grad(c):
        # Normalize c to unit length to handle scale invariance
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Gradients with respect to u
        # d/du (N4) = (1/4) N4^3 * d/du (N4^4) = (1/4) N4^3 * 4 sum(w f^3 A) d_phi = N4^3 sum(w f^3 A) d_phi
        # d/du (N2) = (1/2) N2^-1 * d/du (N2^2) = (1/2) N2^-1 * 2 sum(w f A) d_phi = N2^-1 sum(w f A) d_phi
        
        sum_wf_A = A.T @ (w_flat * f_vals) * d_phi
        sum_wf3_A = A.T @ (w_flat * f_vals**3) * d_phi
        
        grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        # Chain rule: c -> u = c/||c||
        # du/dc = (I - uu^T) / ||c||
        # grad_c R = (I - uu^T) @ grad_u / ||c||
        # (I - uu^T) @ v = v - u (u^T v)
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    # Generate diverse initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Random starts
    for _ in range(20):
        c_rand = rng.standard_normal(n_m)
        candidates.append(c_rand)
        
    # 2. Structured starts based on symmetries
    # Only m=0
    c_m0 = np.zeros(n_m); c_m0[10] = 1.0; candidates.append(c_m0)
    # Only m=10 (cos)
    c_m10c = np.zeros(n_m); c_m10c[20] = 1.0; candidates.append(c_m10c)
    # Only m=10 (sin)
    c_m10s = np.zeros(n_m); c_m10s[0] = 1.0; candidates.append(c_m10s)
    # Only even m
    for m in range(0, 11, 2):
        c_even = np.zeros(n_m)
        c_even[m+10] = 1.0
        if m > 0: c_even[-m+10] = 1.0
        candidates.append(c_even)
    # Only odd m
    for m in range(1, 11, 2):
        c_odd = np.zeros(n_m)
        c_odd[m+10] = 1.0
        c_odd[-m+10] = 1.0
        candidates.append(c_odd)
        
    # 3. Combinations of low m terms
    for m1 in range(0, 11, 2):
        for m2 in range(m1+1, 11, 2):
            c_comb = np.zeros(n_m)
            c_comb[m1+10] = 1.0
            c_comb[m2+10] = 1.0
            if m1 > 0: c_comb[-m1+10] = 1.0
            if m2 > 0: c_comb[-m2+10] = 1.0
            candidates.append(c_comb)

    best_c = None
    best_val = np.inf

    # Optimize each candidate
    for c_init in candidates:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # If time remains, do more random starts
    while time.time() < deadline - 10:
        c_rand = rng.standard_normal(n_m)
        try:
            res = minimize(
                objective_and_grad, 
                c_rand, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Final validity check
    if best_c is None or not np.all(np.isfinite(best_c)):
        return np.ones(n_m)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c
