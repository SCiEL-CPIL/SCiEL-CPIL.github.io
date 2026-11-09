# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv
from scipy.optimize import minimize

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
    k = 10
    start_time = time.time()
    deadline = start_time + 590.0  # Leave 10s margin before 600s limit

    # Set up quadrature points
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    
    # 130 equally spaced phi points
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Precompute sin/cos of m*phi for m = 0..10
    cos_m_phi = np.zeros((11, n_phi))
    sin_m_phi = np.zeros((11, n_phi))
    for m in range(11):
        if m == 0:
            cos_m_phi[m] = 1.0
            sin_m_phi[m] = 0.0
        else:
            cos_m_phi[m] = np.cos(m * phi)
            sin_m_phi[m] = np.sin(m * phi)
    
    # Precompute normalization constants N_m
    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * math.factorial(10 - m) / math.factorial(10 + m))
    
    # Precompute associated Legendre polynomials P_10^m(x) for x in x_gl
    P_legendre = np.zeros((11, 70))
    for m in range(11):
        P_legendre[m] = lpmv(m, 10, x_gl)
    
    # Build the full matrix Y_flat of shape (70*130, 21)
    # Indexing: row = i*n_phi + j, col = m+10
    N_q = 70 * n_phi
    Y_flat = np.zeros((N_q, 21), dtype=np.float64)
    
    # Weights for each quadrature point
    w_flat = np.zeros(N_q, dtype=np.float64)
    
    for i in range(70):
        row_start = i * n_phi
        row_end = (i + 1) * n_phi
        w_flat[row_start:row_end] = w_gl[i] * dphi
        
        # m = 0
        Y_flat[row_start:row_end, 10] = N[0] * P_legendre[0, i]
        
        # m > 0 and m < 0
        for m in range(1, 11):
            coeff_m_i = np.sqrt(2.0) * N[m] * P_legendre[m, i]
            # c_m (index 10+m) corresponds to cos(m*phi)
            Y_flat[row_start:row_end, 10 + m] = coeff_m_i * cos_m_phi[m]
            # c_-m (index 10-m) corresponds to sin(m*phi)
            Y_flat[row_start:row_end, 10 - m] = coeff_m_i * sin_m_phi[m]

    def objective_and_grad(c_vec):
        """
        Compute R = ||f||_4 / ||f||_2 and its gradient.
        c_vec: shape (21,)
        """
        if time.time() > deadline:
            # Return a large value to stop optimization
            return np.inf, np.zeros(21)
            
        f = Y_flat @ c_vec  # shape (N_q,)
        
        f2 = f ** 2
        f4 = f ** 4
        
        norm2_sq = np.sum(w_flat * f2)
        norm4 = np.sum(w_flat * f4)
        
        if norm2_sq <= 1e-300 or norm4 <= 1e-300:
            return np.inf, np.zeros(21)
            
        norm2 = np.sqrt(norm2_sq)
        norm4_val = norm4 ** 0.25
        
        R = norm4_val / norm2
        
        # Gradient
        # R = A^(1/4) B^(-1/2)
        # dR/dc = (1/4) A^(-3/4) B^(-1/2) dA/dc - (1/2) A^(1/4) B^(-3/2) dB/dc
        # dA/dc = 4 sum(w * f^3 * Y)
        # dB/dc = 2 sum(w * f * Y)
        
        wf = w_flat * f
        wf3 = w_flat * f ** 3
        
        dB = 2.0 * (Y_flat.T @ wf)
        dA = 4.0 * (Y_flat.T @ wf3)
        
        A_inv34 = norm4 ** (-0.75)
        B_inv2 = norm2_sq ** (-0.5)
        A_14 = norm4 ** 0.25
        B_inv32 = norm2_sq ** (-1.5)
        
        grad = (0.25 * A_inv34 * B_inv2) * dA - (0.5 * A_14 * B_inv32) * dB
        
        return R, grad

    best_result = None
    best_obj = np.inf
    
    # Define starting points
    # 1. All ones
    starts = []
    starts.append(np.ones(21, dtype=np.float64))
    
    # 2. Only m=0
    c_m0 = np.zeros(21, dtype=np.float64)
    c_m0[10] = 1.0
    starts.append(c_m0)
    
    # 3. Only m=10 (cos)
    c_m10 = np.zeros(21, dtype=np.float64)
    c_m10[20] = 1.0
    starts.append(c_m10)
    
    # 4. Only m=-10 (sin)
    c_mneg10 = np.zeros(21, dtype=np.float64)
    c_mneg10[0] = 1.0
    starts.append(c_mneg10)
    
    # 5. Equal magnitude, alternating signs for "flatness" in phi?
    # Actually, let's just use random starts with fixed seeds for reproducibility
    for seed in range(20):
        np.random.seed(seed)
        c_rand = np.random.randn(21)
        # Normalize to unit norm to avoid scaling issues in initial steps
        c_rand = c_rand / np.linalg.norm(c_rand)
        starts.append(c_rand)
    
    # Optimization loop
    for i, c_start in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
            
        try:
            result = minimize(
                objective_and_grad,
                c_start,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 1000,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxfun': 2000
                }
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x.copy()
                
        except Exception:
            pass
            
    if best_result is None:
        best_result = np.ones(21, dtype=np.float64)
    
    # Polish step: Run a tighter optimization from the best result
    if time.time() < deadline - 5.0:
        try:
            result_polish = minimize(
                objective_and_grad,
                best_result,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 2000,
                    'gtol': 1e-10,
                    'ftol': 1e-14,
                    'maxfun': 5000
                }
            )
            if result_polish.fun < best_obj:
                best_obj = result_polish.fun
                best_result = result_polish.x.copy()
        except Exception:
            pass

    c = best_result
    
    # Final validation
    R_final, _ = objective_and_grad(c)
    if not np.isfinite(R_final):
        c = np.ones(21, dtype=np.float64)
    
    # Ensure valid range
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c / max_abs * 1e-2
    elif max_abs > 1e5:
        c = c / max_abs * 1e4
        
    return c

# EVOLVE-BLOCK-END
