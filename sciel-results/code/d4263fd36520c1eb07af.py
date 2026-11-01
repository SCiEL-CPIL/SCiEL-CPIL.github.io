"""
Uncertainty inequality: minimize A(f)*A(F) for Hermite-Gaussian test functions.
Uses exact Rayleigh quotient structure with generalized eigenvalue analysis
plus high-precision local polishing.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

# Precompute constants
# I[n] = integral x^(2n) e^(-x^2) dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_integrals():
    I = np.zeros(13)
    sp = math.sqrt(math.pi)
    for n in range(13):
        num = math.factorial(2 * n)
        den = (4.0 ** n) * math.factorial(n)
        I[n] = (num / den) * sp
    return I

_I_TABLE = _precompute_integrals()

# Hermite polynomial coefficients (physicists' convention)
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# Coefficient matrix: rows are polynomial degree 0,2,4,6,8,10,12
# Columns correspond to H0, H4, H8, H12
_H_COEFFS = np.array([
    [1.0, 3.0, 105.0, 945.0],       # x^0
    [0.0, -6.0, -420.0, -7560.0],   # x^2
    [0.0, 1.0, 210.0, 13860.0],     # x^4
    [0.0, 0.0, -28.0, -6930.0],     # x^6
    [0.0, 0.0, 1.0, 1155.0],        # x^8
    [0.0, 0.0, 0.0, -66.0],         # x^10
    [0.0, 0.0, 0.0, 1.0],           # x^12
])

# Constraint: g(0) = 0 => c0 + 3c4 + 105c8 + 945c12 = 0
# => c12 = -(c0 + 3c4 + 105c8) / 945
_C12_COEFFS = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

def _compute_p(params):
    """Compute polynomial coefficients p[0..6] from params [c0, c4, c8]."""
    c0, c4, c8 = params
    c12 = _C12_COEFFS[0]*c0 + _C12_COEFFS[1]*c4 + _C12_COEFFS[2]*c8
    coeffs = np.array([c0, c4, c8, c12])
    p = _H_COEFFS @ coeffs
    return p

def _bound_and_grad(params):
    """
    Compute the exact uncertainty bound and its gradient.
    params = [c0, c4, c8]
    """
    p = _compute_p(params)
    
    # Compute moments m0 and m2
    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            if pij == 0.0:
                continue
            idx0 = i + j
            idx2 = i + j + 1
            if idx0 < 13:
                m0 += pij * _I_TABLE[idx0]
            if idx2 < 13:
                m2 += pij * _I_TABLE[idx2]
                
    if m0 <= 0 or not np.isfinite(m0):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    var_x = m2 / m0
    
    if var_x <= 0 or not np.isfinite(var_x):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = var_x  # Since var_x == var_xi due to Fourier symmetry
    
    # Gradient computation
    # dp/dc for each parameter
    dp_dc = np.zeros((7, 3))
    for a in range(3):
        # Direct contribution from c_a
        for col in range(4):
            if col == a:
                dp_dc[:, a] += _H_COEFFS[:, col]
        # Contribution through c12
        dp_dc[:, a] += _H_COEFFS[:, 3] * _C12_COEFFS[a]
        
    grad = np.zeros(3)
    for c in range(3):
        dm0_dc = 0.0
        dm2_dc = 0.0
        for a in range(7):
            if dp_dc[a, c] == 0.0:
                continue
            s0 = 0.0
            s2 = 0.0
            for j in range(7):
                if p[j] == 0.0:
                    continue
                idx0 = a + j
                idx2 = a + j + 1
                if idx0 < 13:
                    s0 += p[j] * _I_TABLE[idx0]
                if idx2 < 13:
                    s2 += p[j] * _I_TABLE[idx2]
            dm0_dc += 2.0 * s0 * dp_dc[a, c]
            dm2_dc += 2.0 * s2 * dp_dc[a, c]
            
        dvar_x = (dm2_dc * m0 - m2 * dm0_dc) / (m0 * m0)
        grad[c] = dvar_x
        
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 170  # Safety margin
    
    # Initial guess from known good solution
    x0 = np.array([-1.13, 0.04, 0.000308642])
    
    best_x = x0.copy()
    best_score = _objective(x0)
    
    # Strategy 1: Multi-start local optimization with L-BFGS-B
    # Use diverse starting points to avoid local minima
    np.random.seed(42)
    
    # Generate diverse starting points
    starts = [x0.copy()]
    
    # Random starts in reasonable ranges
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        x_start = np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.05, 0.05)
        ])
        starts.append(x_start)
    
    # Also try structured starts
    for c0_val in [-2, -1, 0, 1, 2]:
        for c4_val in [-0.1, 0, 0.1]:
            for c8_val in [-0.01, 0, 0.01]:
                x_start = np.array([c0_val, c4_val, c8_val])
                starts.append(x_start)
    
    # Run L-BFGS-B from each start
    for x_start in starts:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(_objective_grad, x_start, jac=True, method='L-BFGS-B',
                          options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Differential Evolution for global search
    if time.time() < deadline - 50:
        try:
            bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
            res_de = minimize(
                _objective, 
                x0, 
                bounds=bounds, 
                method='differential_evolution',
                options={
                    'maxiter': 80, 
                    'polish': True, 
                    'init': 'sobol',
                    'popsize': 12,
                    'seed': 42,
                    'tol': 1e-8
                }
            )
            if res_de.fun < best_score and np.isfinite(res_de.fun):
                best_score = res_de.fun
                best_x = res_de.x.copy()
        except Exception:
            pass
    
    # Strategy 3: High-precision polish from best candidate
    if time.time() < deadline - 10:
        try:
            res_final = minimize(_objective_grad, best_x, jac=True, method='L-BFGS-B',
                                 options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if res_final.fun < best_score and np.isfinite(res_final.fun):
                best_score = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass
    
    # Strategy 4: Try to use the Rayleigh quotient structure
    # The objective is m2/m0 where m0 and m2 are quadratic forms in p
    # p is linear in params, so this is a ratio of quadratic forms
    # We can try to find stationary points by solving the generalized eigenvalue problem
    
    if time.time() < deadline - 5:
        # Build the matrices A and B such that m_k = p^T M_k p
        # where M_k[i,j] = I[k+i+j]
        M0 = np.zeros((7, 7))
        M2 = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                idx0 = i + j
                idx2 = i + j + 1
                if idx0 < 13:
                    M0[i, j] = _I_TABLE[idx0]
                if idx2 < 13:
                    M2[i, j] = _I_TABLE[idx2]
        
        # p = C @ params where C is 7x4 (including c12)
        # But params are only 3 (c0, c4, c8), c12 is determined
        # Let's work in the 3-parameter space
        # p = D @ params where D is 7x3
        D = np.zeros((7, 3))
        for a in range(3):
            for col in range(4):
                if col == a:
                    D[:, a] += _H_COEFFS[:, col]
            D[:, a] += _H_COEFFS[:, 3] * _C12_COEFFS[a]
        
        # m0 = params^T (D^T M0 D) params
        # m2 = params^T (D^T M2 D) params
        A = D.T @ M0 @ D  # 3x3 matrix for m0
        B = D.T @ M2 @ D  # 3x3 matrix for m2
        
        # Find stationary points of B(x)/A(x)
        # These satisfy B x = lambda A x
        # Solve generalized eigenvalue problem
        try:
            # Use scipy.linalg.eigh for symmetric definite case
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(B, A)
            
            # For each eigenvalue, check if it's a valid point (m0 > 0)
            for i in range(len(eigenvalues)):
                if time.time() > deadline - 2:
                    break
                x_candidate = eigenvectors[:, i]
                # Normalize to have reasonable magnitude
                norm = np.linalg.norm(x_candidate)
                if norm > 0:
                    x_candidate = x_candidate / norm
                    # Try both signs
                    for sign in [1, -1]:
                        x_test = sign * x_candidate
                        try:
                            val, _ = _bound_and_grad(x_test)
                            if val < best_score and np.isfinite(val):
                                best_score = val
                                best_x = x_test.copy()
                        except Exception:
                            pass
        except Exception:
            pass
    
    # Final check and return
    if not np.isfinite(best_score) or best_score > 1e10:
        best_x = x0.copy()
        best_score = _objective(x0)
    
    return best_x

# EVOLVE-BLOCK-END
