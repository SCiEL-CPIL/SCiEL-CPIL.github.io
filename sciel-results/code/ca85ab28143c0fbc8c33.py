"""Optimized program for uncertainty inequality task using generalized eigenvalue approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigh
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


def _build_matrices():
    """
    Build matrices A and B for the quadratic forms after substituting c12.
    
    The test function is g(x) = P(x) * exp(-x^2/2) where P(x) is a polynomial.
    We have P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    with constraint P(0) = 0 determining c12.
    
    After substitution, we get a 3-variable problem with quadratic forms.
    """
    # H0(0)=1, H4(0)=-12, H8(0)=2520, H12(0)=-103950
    # c0 + c4*(-12) + c8*2520 + c12*(-103950) = 0
    # c12 = (c0 - 12*c4 + 2520*c8) / 103950.0
    
    D = 103950.0
    inv_D = 1.0 / D
    
    # P(x) = sum_{k=0}^{6} p_k * x^(2k)
    # where p_k depends linearly on c = [c0, c4, c8]
    
    # Coefficients for each power of x^(2k):
    # p[0] (x^0): c0*1 + c4*3 + c8*105 + c12*103950
    #   = c0 + 3*c4 + 105*c8 + (c0 - 12*c4 + 2520*c8)
    #   = 2*c0 - 9*c4 + 2625*c8
    p0 = np.array([2.0, -9.0, 2625.0])
    
    # p[1] (x^2): c4*(-6) + c8*(-1680) + c12*(-3991680)
    #   = -6*c4 - 1680*c8 - 3991680*(c0 - 12*c4 + 2520*c8)/103950
    p1 = np.array([
        -3991680.0 * inv_D,
        -6.0 + 3991680.0 * 12.0 * inv_D,
        -1680.0 - 3991680.0 * 2520.0 * inv_D
    ])
    
    # p[2] (x^4): c4*1 + c8*420 + c12*1188000
    p2 = np.array([
        1188000.0 * inv_D,
        1.0 - 1188000.0 * 12.0 * inv_D,
        420.0 + 1188000.0 * 2520.0 * inv_D
    ])
    
    # p[3] (x^6): c8*(-28) + c12*(-118800)
    p3 = np.array([
        -118800.0 * inv_D,
        118800.0 * 12.0 * inv_D,
        -28.0 - 118800.0 * 2520.0 * inv_D
    ])
    
    # p[4] (x^8): c8*1 + c12*3960
    p4 = np.array([
        3960.0 * inv_D,
        -3960.0 * 12.0 * inv_D,
        1.0 + 3960.0 * 2520.0 * inv_D
    ])
    
    # p[5] (x^10): c12*(-66)
    p5 = np.array([
        -66.0 * inv_D,
        66.0 * 12.0 * inv_D,
        -66.0 * 2520.0 * inv_D
    ])
    
    # p[6] (x^12): c12*1
    p6 = np.array([
        inv_D,
        -12.0 * inv_D,
        2520.0 * inv_D
    ])
    
    # P_mat: 7x3 matrix where p_vec = P_mat @ c
    P_mat = np.vstack([p0, p1, p2, p3, p4, p5, p6])
    
    # For the Gaussian-weighted inner product:
    # <f, g> = ∫ f(x) g(x) exp(-x^2) dx
    # If f(x) = sum_i a_i x^(2i) exp(-x^2/2), g(x) = sum_j b_j x^(2j) exp(-x^2/2)
    # Then <f, g> = sum_{i,j} a_i b_j ∫ x^(2i+2j) exp(-x^2) dx = sum_{i,j} a_i b_j I(i+j)
    
    # norm_sq = ||g||^2 = c^T B c where B[i,j] = sum_{m,n} P_mat[m,i] * P_mat[n,j] * I(m+n)
    # x2_norm = ||x*g||^2 = c^T A c where A[i,j] = sum_{m,n} P_mat[m,i] * P_mat[n,j] * I(m+n+1)
    
    M_I = np.zeros((7, 7))
    M_I1 = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            M_I[i, j] = _I_CACHE[i + j]
            M_I1[i, j] = _I_CACHE[i + j + 1]
    
    B_mat = P_mat.T @ M_I @ P_mat
    A_mat = P_mat.T @ M_I1 @ P_mat
    
    return A_mat, B_mat


# Precompute the matrices
_A_mat, _B_mat = _build_matrices()


def _compute_raw_value(c0, c4, c8):
    """Compute the raw uncertainty product for given coefficients."""
    c = np.array([c0, c4, c8])
    norm_sq = c @ _B_mat @ c
    x2_norm = c @ _A_mat @ c
    
    if norm_sq <= 1e-30:
        return 1e10
    
    return x2_norm / norm_sq


def _compute_gradient(c0, c4, c8):
    """Compute the gradient of the objective."""
    c = np.array([c0, c4, c8])
    norm_sq = c @ _B_mat @ c
    x2_norm = c @ _A_mat @ c
    
    if norm_sq <= 1e-30:
        return np.zeros(3)
    
    # d/dc (c^T A c / c^T B c) = (2*A_mat@c*norm_sq - (c^T A c)*2*B_mat@c) / (norm_sq)^2
    Ac = _A_mat @ c
    Bc = _B_mat @ c
    grad = (2.0 * Ac * norm_sq - 2.0 * x2_norm * Bc) / (norm_sq ** 2)
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s limit
    
    # Seed values
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    # Verify seed
    seed_val = _compute_raw_value(*seed)
    
    # Phase 1: Solve the generalized eigenvalue problem directly
    # The minimum of (c^T A c) / (c^T B c) over c != 0 is the smallest generalized eigenvalue
    # of the pair (A, B), i.e., the smallest λ such that A c = λ B c
    best_x = seed.copy()
    best_val = seed_val
    
    try:
        # Use eigh for symmetric definite generalized eigenvalue problem
        # B should be positive definite for the norm
        # Check if B is positive definite
        eig_B = np.linalg.eigvalsh(_B_mat)
        
        if np.all(eig_B > 1e-10):
            # B is positive definite, solve A c = λ B c
            eigvals, eigvecs = eigh(_A_mat, _B_mat)
            
            # The smallest eigenvalue gives the minimum
            min_idx = np.argmin(eigvals)
            eigvec = eigvecs[:, min_idx]
            
            # Normalize: the ratio is scale-invariant, so we can use any scaling
            # But we need to check if this gives a valid (non-zero norm) solution
            test_val = _compute_raw_value(*eigvec)
            
            if test_val < best_val and test_val > 0:
                best_val = test_val
                best_x = eigvec.copy()
                
                # Polish with local optimization
                try:
                    def objective_with_grad(x):
                        val = _compute_raw_value(x[0], x[1], x[2])
                        grad = _compute_gradient(x[0], x[1], x[2])
                        return val, grad
                    
                    result = minimize(objective_with_grad, best_x, method='L-BFGS-B',
                                     options={'maxiter': 10000, 'gtol': 1e-16, 'ftol': 1e-16})
                    if result.fun < best_val and result.fun > 0:
                        best_val = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass
    except Exception:
        pass
    
    if time.time() > deadline - 60:
        return best_x
    
    # Phase 2: Multi-start Nelder-Mead for robustness
    rng = np.random.default_rng(123)
    
    def objective(x):
        return _compute_raw_value(x[0], x[1], x[2])
    
    # Start from the eigenvalue solution and seed
    starts = [best_x.copy(), seed.copy()]
    
    # Add some random perturbations
    for _ in range(10):
        starts.append(best_x + rng.normal(0, 0.5, size=3))
        starts.append(seed + rng.normal(0, 1.0, size=3))
    
    for x0 in starts:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if 0 < result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if time.time() > deadline - 15:
        return best_x
    
    # Phase 3: Final high-precision polish
    try:
        def objective_with_grad(x):
            val = _compute_raw_value(x[0], x[1], x[2])
            grad = _compute_gradient(x[0], x[1], x[2])
            return val, grad
        
        result = minimize(objective_with_grad, best_x, method='L-BFGS-B',
                         options={'maxiter': 20000, 'gtol': 1e-18, 'ftol': 1e-18})
        if 0 < result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
