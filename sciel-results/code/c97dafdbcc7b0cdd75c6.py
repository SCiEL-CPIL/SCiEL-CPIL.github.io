import numpy as np
from scipy.special import lpmv, factorial
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A on the specified quadrature grid.
    2. Use fixed-point iteration on the Euler-Lagrange condition:
       f ~ f^4 (for L2-normalized f).
       Specifically, update f_{k+1} = A (A^T (f_k^4)) / ||A (A^T (f_k^4))||_2
       This is a power iteration for the non-linear eigenvalue problem.
       To improve convergence, we use over-relaxation and multiple random starts.
    3. Return the best valid coefficient vector.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

    # 1. Setup Quadrature Grid
    mu, w = np.polynomial.legendre.leggauss(70)
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
        
    # Precompute weighted A^T for efficient updates
    # We need to compute c_new ~ A^T (w * (A c)^4)
    # Let M = A^T diag(w) A. This is not linear, so we can't precompute M simply.
    # But we can precompute W = diag(w).
    # c_new = A^T (W (A c)^4)
    
    # Objective for evaluation
    def evaluate(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return np.inf, np.zeros(n_m)
        u = c / norm_c
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        
        if norm2_sq < 1e-12:
            return np.inf, u
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        return R, u

    best_c = None
    best_val = np.inf
    
    # 3. Fixed Point Iteration
    # Start with several random initializations
    rng = np.random.default_rng(42)
    
    # Also include some structured starts (e.g. pure m=0, pure m=10)
    starts = []
    for _ in range(50):
        c_start = rng.standard_normal(n_m)
        c_start /= np.linalg.norm(c_start)
        starts.append(c_start)
        
    # Add pure modes
    for m in range(0, 11):
        c_pure = np.zeros(n_m)
        if m == 0:
            c_pure[10] = 1.0
        else:
            c_pure[m + 10] = 1.0 # Cosine part
        starts.append(c_pure)
        c_pure2 = np.zeros(n_m)
        c_pure2[-m + 10] = 1.0 # Sine part
        starts.append(c_pure2)

    # Relaxation parameter. 
    # The map is T(c) = A (A^T (A c)^4) normalized.
    # We can try mixing: c_new = (1-alpha) c_old + alpha T(c_old)
    # Or just plain power iteration.
    
    alpha = 0.5
    max_iter = 200
    
    for c_init in starts:
        if time.time() > deadline - 10:
            break
            
        c_curr = c_init.copy()
        c_curr /= np.linalg.norm(c_curr)
        
        f_curr = A @ c_curr
        f_curr4 = f_curr ** 4
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Compute T(c)
            # g = w * f^4
            g = w_flat * f_curr4
            # c_proj = A^T g * d_phi
            c_proj = A.T @ g * d_phi
            c_proj /= np.linalg.norm(c_proj)
            
            # Update
            c_next = (1 - alpha) * c_curr + alpha * c_proj
            c_next /= np.linalg.norm(c_next)
            
            # Check convergence
            if np.linalg.norm(c_next - c_curr) < 1e-10:
                c_curr = c_next
                break
            else:
                c_curr = c_next
                f_curr = A @ c_curr
                f_curr4 = f_curr ** 4
                
        # Evaluate result
        val, u_final = evaluate(c_curr)
        if val < best_val:
            best_val = val
            best_c = u_final.copy()
            
    # Fallback if all failed
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    # Final Validity Check and Scaling
    if not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c
