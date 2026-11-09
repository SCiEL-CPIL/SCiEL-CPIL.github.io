"""Rewritten program with stronger optimization for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.linalg import eigvals
import math


def _I(n):
    """I(n) = ∫_{-∞}^{∞} x^(2n) exp(-x^2) dx = sqrt(pi) * (2n-1)!! / 2^n"""
    if n == 0:
        return np.sqrt(np.pi)
    result = 1.0
    for k in range(1, 2*n, 2):
        result *= k
    return np.sqrt(np.pi) * result / (2.0 ** n)


# Precompute I(0) through I(14) for speed
_I_CACHE = [float(_I(n)) for n in range(15)]


def _build_quadratic_forms():
    """
    Build matrices A and B such that:
    norm_sq = c^T B c  (where c = [c0, c4, c8, c12] but c12 is determined by constraint)
    x2_norm = c^T A c
    
    Actually, since c12 is determined by c0, c4, c8 via the constraint g(0)=0,
    we can substitute and get a 3-variable quadratic form.
    
    Let's compute the objective as a function of (c0, c4, c8) directly.
    """
    # H0(0)=1, H4(0)=-12, H8(0)=2520, H12(0)=-103950
    # c0 + c4*(-12) + c8*2520 + c12*(-103950) = 0
    # c12 = (c0 - 12*c4 + 2520*c8) / 103950.0
    
    # P(x) coefficients in terms of x^(2i):
    # p[0] = c0 + 3*c4 + 105*c8 + 103950*c12
    # p[1] = -6*c4 - 1680*c8 - 3991680*c12
    # p[2] = c4 + 420*c8 + 1188000*c12
    # p[3] = -28*c8 - 118800*c12
    # p[4] = c8 + 3960*c12
    # p[5] = -66*c12
    # p[6] = c12
    
    # Let's express p in terms of c = [c0, c4, c8]:
    # c12 = (c0 - 12*c4 + 2520*c8) / 103950.0
    # Let D = 103950.0
    # c12 = c0/D - 12*c4/D + 2520*c8/D
    
    D = 103950.0
    inv_D = 1.0 / D
    
    # p[0] = c0 + 3*c4 + 105*c8 + 103950*(c0/D - 12*c4/D + 2520*c8/D)
    #      = c0 + 3*c4 + 105*c8 + c0 - 12*c4 + 2520*c8
    #      = 2*c0 - 9*c4 + 2625*c8
    # p[1] = -6*c4 - 1680*c8 - 3991680*(c0/D - 12*c4/D + 2520*c8/D)
    #      = -6*c4 - 1680*c8 - 3991680/D*c0 + 3991680*12/D*c4 - 3991680*2520/D*c8
    # 3991680/D = 3991680/103950 ≈ 38.398...
    # Let's compute exactly:
    p0_c0 = 2.0
    p0_c4 = -9.0
    p0_c8 = 2625.0
    
    p1_c0 = -3991680.0 * inv_D
    p1_c4 = -6.0 + 3991680.0 * 12.0 * inv_D
    p1_c8 = -1680.0 - 3991680.0 * 2520.0 * inv_D
    
    p2_c0 = 1188000.0 * inv_D
    p2_c4 = 1.0 - 1188000.0 * 12.0 * inv_D
    p2_c8 = 420.0 + 1188000.0 * 2520.0 * inv_D
    
    p3_c0 = -118800.0 * inv_D
    p3_c4 = 118800.0 * 12.0 * inv_D
    p3_c8 = -28.0 - 118800.0 * 2520.0 * inv_D
    
    p4_c0 = 3960.0 * inv_D
    p4_c4 = -3960.0 * 12.0 * inv_D
    p4_c8 = 1.0 + 3960.0 * 2520.0 * inv_D
    
    p5_c0 = -66.0 * inv_D
    p5_c4 = 66.0 * 12.0 * inv_D
    p5_c8 = -66.0 * 2520.0 * inv_D
    
    p6_c0 = inv_D
    p6_c4 = -12.0 * inv_D
    p6_c8 = 2520.0 * inv_D
    
    # Build matrix P such that p_vec = P @ c where c = [c0, c4, c8]
    P_mat = np.array([
        [p0_c0, p0_c4, p0_c8],
        [p1_c0, p1_c4, p1_c8],
        [p2_c0, p2_c4, p2_c8],
        [p3_c0, p3_c4, p3_c8],
        [p4_c0, p4_c4, p4_c8],
        [p5_c0, p5_c4, p5_c8],
        [p6_c0, p6_c4, p6_c8]
    ])
    
    # Now q_m = sum_{i+j=m} p_i * p_j for m=0..12
    # q = (p_vec ⊗ p_vec) rearranged
    # norm_sq = sum_m q_m * I(m) = sum_{i,j} p_i p_j I(i+j)
    # x2_norm = sum_m q_m * I(m+1) = sum_{i,j} p_i p_j I(i+j+1)
    
    # So norm_sq = c^T (P^T M_I P) c where M_I[i,j] = I(i+j)
    # And x2_norm = c^T (P^T M_I1 P) c where M_I1[i,j] = I(i+j+1)
    
    M_I = np.zeros((7, 7))
    M_I1 = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            M_I[i, j] = _I_CACHE[i + j]
            M_I1[i, j] = _I_CACHE[i + j + 1]
    
    B_mat = P_mat.T @ M_I @ P_mat  # norm_sq = c^T B_mat c
    A_mat = P_mat.T @ M_I1 @ P_mat  # x2_norm = c^T A_mat c
    
    return A_mat, B_mat


# Precompute the quadratic form matrices
_A_mat, _B_mat = _build_quadratic_forms()


def _compute_raw_value(c0, c4, c8):
    """Compute the raw uncertainty product A(g)*A(F(g)) for the Hermite-Gaussian test function."""
    c = np.array([c0, c4, c8])
    norm_sq = c @ _B_mat @ c
    x2_norm = c @ _A_mat @ c
    
    if norm_sq <= 0:
        return 1e10
    
    return x2_norm / norm_sq


def _compute_gradient(c0, c4, c8):
    """Compute the gradient of the objective for Newton-type methods."""
    c = np.array([c0, c4, c8])
    norm_sq = c @ _B_mat @ c
    x2_norm = c @ _A_mat @ c
    
    if norm_sq <= 0:
        return np.zeros(3)
    
    # d/dc (x2_norm / norm_sq) = (2*A_mat@c*norm_sq - x2_norm*2*B_mat@c) / norm_sq^2
    grad_num = 2.0 * (_A_mat @ c) * norm_sq - x2_norm * 2.0 * (_B_mat @ c)
    grad = grad_num / (norm_sq ** 2)
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s limit
    
    # Seed values
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    # Verify seed gives ~1.0
    seed_val = _compute_raw_value(*seed)
    
    # If our formula doesn't give ~1.0 for the seed, something is wrong
    if abs(seed_val - 1.0) > 0.1:
        return seed
    
    # Phase 1: Multiple random restarts with Nelder-Mead
    best_x = seed.copy()
    best_val = seed_val
    
    def objective(x):
        return _compute_raw_value(x[0], x[1], x[2])
    
    # Try several random starting points
    rng = np.random.default_rng(42)
    
    for trial in range(20):
        if time.time() > deadline - 90:
            break
        
        # Generate random starting point
        if trial == 0:
            x0 = seed.copy()
        else:
            x0 = seed + rng.normal(0, 1.0, size=3)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if time.time() > deadline - 60:
        return best_x
    
    # Phase 2: Differential Evolution for global exploration
    if time.time() < deadline - 45:
        try:
            bounds = [(-10, 10), (-10, 10), (-10, 10)]
            de_result = differential_evolution(objective, bounds, seed=42, 
                                               maxiter=50, tol=1e-12,
                                               polish=True, init='sobol',
                                               popsize=15)
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass
    
    if time.time() > deadline - 30:
        return best_x
    
    # Phase 3: High-precision Newton-type polish using exact gradient
    # Use BFGS with very tight tolerance
    try:
        def objective_with_grad(x):
            val = _compute_raw_value(x[0], x[1], x[2])
            grad = _compute_gradient(x[0], x[1], x[2])
            return val, grad
        
        result = minimize(objective_with_grad, best_x, method='BFGS',
                         options={'maxiter': 10000, 'gtol': 1e-16})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 15:
        return best_x
    
    # Phase 4: L-BFGS-B with very tight tolerance for final polish
    try:
        def objective_with_grad2(x):
            val = _compute_raw_value(x[0], x[1], x[2])
            grad = _compute_gradient(x[0], x[1], x[2])
            return val, grad
        
        result = minimize(objective_with_grad2, best_x, method='L-BFGS-B',
                         options={'maxiter': 20000, 'gtol': 1e-18, 'ftol': 1e-18})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        return best_x
    
    # Phase 5: Final ultra-tight Nelder-Mead polish
    try:
        result = minimize(objective, best_x, method='Nelder-Mead',
                         options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
