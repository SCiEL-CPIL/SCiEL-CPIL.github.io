import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution
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
    # Gauss-Legendre nodes and weights for cos(theta)
    x, w_theta = np.polynomial.legendre.leggauss(N_THETA)
    # x is in [-1, 1], representing cos(theta)
    # The integral over theta is int_0^pi f(theta) sin(theta) dtheta = int_{-1}^1 f(arccos(x)) dx
    # So the weight for the cos(theta) grid is just w_theta.
    # Wait, the measure is dOmega = sin(theta) dtheta dphi = dx dphi.
    # So the weights for the sum over theta are w_theta.
    
    phi = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    
    # Total number of points
    N_POINTS = N_THETA * N_PHI
    
    # 2. Precompute Spherical Harmonics Y_m at quadrature points
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    # N_m = sqrt( (2K+1)/(4 pi) * (K-m)! / (K+m)! )
    
    Y = np.zeros((N_POINTS, 2 * K + 1))
    
    # Flatten grids
    # x is (N_THETA,), phi is (N_PHI,)
    # We need to evaluate for each (i, j)
    
    # Compute N_m
    N_m = np.zeros(K + 1)
    for m in range(K + 1):
        # Factorials can be large, use log or direct float
        # (20)! is ~ 2e18, fits in float64
        num = (2 * K + 1) / (4 * np.pi) * factorial(K - m) / factorial(K + m)
        N_m[m] = np.sqrt(num)
    
    # Evaluate P_10^m(x)
    # lpmv(m, l, x) includes Condon-Shortley phase
    P = np.zeros((K + 1, N_THETA))
    for m in range(K + 1):
        P[m] = lpmv(m, K, x)
        
    # Fill Y matrix
    # Index mapping: m = -10..10 -> index 0..20
    # m_idx = m + 10
    
    # Reshape for broadcasting
    # x is (N_THETA, 1), phi is (1, N_PHI)
    x_grid = x[:, np.newaxis]
    phi_grid = phi[np.newaxis, :]
    
    for m in range(K + 1):
        idx_pos = m + K
        idx_neg = -m + K
        
        N_val = N_m[m]
        P_val = P[m, :, np.newaxis] # (N_THETA, 1)
        
        if m == 0:
            # Y_0 = N_0 P_10^0(cos theta)
            # Shape (N_THETA, 1) needs to be broadcast to (N_THETA, N_PHI)
            Y_m = N_val * P_val * np.ones((1, N_PHI))
            Y[:, idx_pos] = Y_m.flatten()
        else:
            cos_m_phi = np.cos(m * phi_grid) # (1, N_PHI)
            sin_m_phi = np.sin(m * phi_grid) # (1, N_PHI)
            
            # Y_pos = sqrt(2) * N_m * P_m * cos(m phi)
            Y_pos = np.sqrt(2.0) * N_val * P_val * cos_m_phi # (N_THETA, N_PHI)
            Y_neg = np.sqrt(2.0) * N_val * P_val * sin_m_phi # (N_THETA, N_PHI)
            
            Y[:, idx_pos] = Y_pos.flatten()
            Y[:, idx_neg] = Y_neg.flatten()
            
    # Weights for the double sum
    # Sum_i,j w_theta[i] * (1/N_PHI) * Y[i,j]^p
    # The total measure is 4 pi.
    # int dOmega = int_{-1}^1 dx int_0^{2pi} dphi = 2 * 2pi = 4pi.
    # Quadrature: sum_i w_theta[i] * sum_j (2pi / N_PHI) * f(x_i, phi_j)
    w_phi = 2 * np.pi / N_PHI
    weights = np.kron(w_theta, np.ones(N_PHI) * w_phi) # (N_POINTS,)
    
    # 3. Define Objective and Gradient
    def objective_and_grad(c):
        # c is shape (21,)
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

    # 4. Optimization: Differential Evolution + Local Polish
    # The objective R = ||f||_4/||f||_2 is scale-invariant and non-convex.
    # Use DE for global search, then L-BFGS-B for local refinement.
    
    # Good initial guess: m=0 only (index K)
    c_init = np.zeros(2 * K + 1, dtype=np.float64)
    c_init[K] = 1.0
    
    best_c = c_init.copy()
    best_val = objective_and_grad(best_c)[0]
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Differential Evolution (global search)
    # DE works on bounded box; we normalize c after to handle scale invariance
    bounds = [(-1.0, 1.0)] * (2 * K + 1)
    
    def de_objective(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-10:
            return 1e10
        c_norm = c / norm_c
        return objective_and_grad(c_norm)[0]
    
    # Check time before DE
    if time.time() < DEADLINE - SAFETY_MARGIN - 120:
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            c_de = de_result.x
            norm_de = np.linalg.norm(c_de)
            if norm_de > 1e-10:
                c_de = c_de / norm_de
                val_de = objective_and_grad(c_de)[0]
                if val_de < best_val:
                    best_val = val_de
                    best_c = c_de
        except Exception:
            pass
    
    # Phase 2: Local polish from multiple starts
    starts = [best_c.copy(), c_init.copy()]
    
    # Add random starts on the unit sphere
    for _ in range(15):
        c_rand = rng.normal(size=2 * K + 1)
        norm_r = np.linalg.norm(c_rand)
        if norm_r > 1e-10:
            starts.append(c_rand / norm_r)
    
    for c0 in starts:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
            
        norm_c0 = np.linalg.norm(c0)
        if norm_c0 < 1e-10:
            continue
        c0 = c0 / norm_c0
        
        try:
            res = minimize(
                lambda c: objective_and_grad(c)[0],
                c0,
                jac=lambda c: objective_and_grad(c)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-12}
            )
            
            c_opt = res.x
            norm_opt = np.linalg.norm(c_opt)
            if norm_opt > 1e-10:
                c_opt = c_opt / norm_opt
                val_opt = objective_and_grad(c_opt)[0]
                
                if val_opt < best_val:
                    best_val = val_opt
                    best_c = c_opt
        except Exception:
            continue
    
    # Phase 3: High-precision final polish
    if time.time() < DEADLINE - SAFETY_MARGIN:
        try:
            norm_best = np.linalg.norm(best_c)
            if norm_best > 1e-10:
                best_c_norm = best_c / norm_best
                res_final = minimize(
                    lambda c: objective_and_grad(c)[0],
                    best_c_norm,
                    jac=lambda c: objective_and_grad(c)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'gtol': 1e-14}
                )
                c_final = res_final.x
                norm_final = np.linalg.norm(c_final)
                if norm_final > 1e-10:
                    c_final = c_final / norm_final
                    val_final = objective_and_grad(c_final)[0]
                    if val_final < best_val:
                        best_val = val_final
                        best_c = c_final
        except Exception:
            pass
            
    # Final check
    # Ensure validity: no NaN, inf, max|c| in [1e-3, 1e5]
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(2 * K + 1, dtype=np.float64)
        best_c[K] = 1.0
    
    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3:
        best_c = best_c * (1.0 / max_abs_c)
    elif max_abs_c > 1e5:
        best_c = best_c * (1e5 / max_abs_c)
        
    return best_c
