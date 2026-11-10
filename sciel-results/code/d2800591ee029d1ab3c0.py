import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy: 
    1. Precompute basis matrix A on quadrature grid.
    2. Define objective R(c) and gradient.
    3. Since R is scale invariant, optimize on the unit sphere ||c||=1.
       We use L-BFGS-B on the coefficients directly but normalize periodically 
       or just rely on the fact that L-BFGS-B handles the valley well if initialized well.
       Actually, simpler: just minimize R(c) directly. The gradient points towards 
       lower R. Scale invariance means the gradient is orthogonal to c at the optimum? 
       No, dR/dc is well-defined.
       
       Better approach: 
       Use multiple initial candidates generated from structured patterns.
       Run L-BFGS-B on each. Keep the best.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin

    k = 10
    n_m = 2 * k + 1 # 21 coefficients
    
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

    # 3. Objective Function and Gradient
    def objective_and_grad(c):
        # c is shape (21,)
        f_vals = A @ c
        f2 = f_vals ** 2
        f4 = f2 ** 2
        
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Gradients
        # dR/dc = (N4^3/N2) * sum w f^3 A^T - (N4/N2^3) * sum w f A^T
        w_f = w_flat * f_vals
        w_f3 = w_flat * f_vals ** 3
        
        sum_wf_A = A.T @ w_f * d_phi
        sum_wf3_A = A.T @ w_f3 * d_phi
        
        grad = (N4**3 / N2) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        return R, grad

    # 4. Generate Initial Candidates
    # We want diverse starts.
    # 1. All ones
    # 2. Single modes (m=0, m=1..10 cos, m=1..10 sin)
    # 3. Random sparse vectors
    # 4. Low frequency bias (higher m smaller)
    
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(n_m))
    
    # 2. Single modes
    for i in range(n_m):
        c = np.zeros(n_m)
        c[i] = 1.0
        candidates.append(c)
    
    # 3. Random sparse (L1-ish)
    rng = np.random.default_rng(42)
    for _ in range(20):
        c = rng.normal(size=n_m)
        # Make it sparse by thresholding
        c[np.abs(c) < np.median(np.abs(c))] = 0
        if np.max(np.abs(c)) > 0:
            c = c / np.max(np.abs(c))
        candidates.append(c)
        
    # 4. Frequency dependent
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        m_vals = np.arange(-k, k+1)
        c = 1.0 / (1.0 + alpha * np.abs(m_vals))
        candidates.append(c)
        
    # 5. Pure m=10 (highest frequency)
    c10 = np.zeros(n_m)
    c10[0] = 1.0 # m=-10
    c10[-1] = 1.0 # m=10
    candidates.append(c10)
    
    # 6. Pure m=0
    c0 = np.zeros(n_m)
    c0[10] = 1.0
    candidates.append(c0)

    best_c = candidates[0]
    best_val = np.inf
    
    # 5. Optimize each candidate
    # We have 600s. 
    # L-BFGS-B is fast. Let's allocate time per candidate.
    # There are ~40 candidates. 600/40 = 15s each.
    # But we can stop early if time is running out.
    
    for i, c0 in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        try:
            # Normalize initial guess to unit norm to avoid scaling issues in early iterations?
            # R is scale invariant, but L-BFGS-B might wander in scale.
            # It's often better to fix scale or let it be. 
            # Let's just pass it.
            res = minimize(
                objective_and_grad, 
                c0, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-10}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
                
        except Exception:
            pass

    # 6. Refine the best candidate with a second pass, potentially using more iterations or different method
    if time.time() < deadline - 30 and np.all(np.isfinite(best_c)):
        try:
            res2 = minimize(
                objective_and_grad, 
                best_c, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-12}
            )
            if res2.fun < best_val:
                best_c = res2.x
                best_val = res2.fun
        except Exception:
            pass

    # Final check on validity
    if not np.all(np.isfinite(best_c)):
        return np.ones(n_m)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Normalize if out of bounds?
        # The problem says invalid if max|c| < 1e-3 or > 1e5.
        # Since R is scale invariant, we can scale c to be within bounds.
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c

# EVOLVE-BLOCK-END
