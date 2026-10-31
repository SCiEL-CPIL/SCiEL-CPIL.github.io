import numpy as np
import time
from scipy.optimize import minimize

# Set a hard deadline with a safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 190  # 20s margin before the 210s kill

# Precompute integrals I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
# I[n] = (2n)! / (4^n n!) * sqrt(pi)
# We need n up to 12 (since max degree is 12, product is 24, so 2n=24 => n=12)
I = np.zeros(13)
I[0] = np.sqrt(np.pi)
for n in range(1, 13):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Hermite polynomial coefficients in power basis x^{2k}
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

H0_c = np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)
H4_c = np.array([3, -6, 1, 0, 0, 0, 0], dtype=float)
H8_c = np.array([105, -420, 210, -28, 1, 0, 0], dtype=float)
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1], dtype=float)

# Constraint: g(0) = 0 => a0 + 3*a4 + 105*a8 + 945*a12 = 0
# a12 = -(a0 + 3*a4 + 105*a8) / 945
# Coefficient vector for a12 in terms of [a0, a4, a8]
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0], dtype=float)

# Build the matrix A such that p = A @ a
# p_k = sum_m A[k,m] a_m
# A[k, 0] = H0_c[k] + c12[0]*H12_c[k]
# A[k, 1] = H4_c[k] + c12[1]*H12_c[k]
# A[k, 2] = H8_c[k] + c12[2]*H12_c[k]
A_mat = np.column_stack([
    H0_c + c12[0]*H12_c,
    H4_c + c12[1]*H12_c,
    H8_c + c12[2]*H12_c
])

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    a: array of shape (3,)
    Returns: (bound, grad)
    """
    p = A_mat @ a
    
    # q_k = p_k * (-1)^k
    signs = np.array([1, -1, 1, -1, 1, -1, 1], dtype=float)
    q = p * signs
    
    # Compute moments Mx[0], Mx[1], Mxi[0], Mxi[1]
    # Mx[0] = sum_{i,j} p_i p_j I[i+j]
    # Mx[1] = sum_{i,j} p_i p_j I[i+j+1]
    # Mxi[0] = sum_{i,j} q_i q_j I[i+j]
    # Mxi[1] = sum_{i,j} q_i q_j I[i+j+1]
    
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            pj = p[j]
            qj = q[j]
            Mx0 += pi * pj * I[i+j]
            Mx1 += pi * pj * I[i+j+1]
            Mxi0 += qi * qj * I[i+j]
            Mxi1 += qi * qj * I[i+j+1]
            
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation
    # dM[n]_m = 2 * sum_i A[i,m] * sum_j p_j I[i+j+n]
    Sx0 = np.zeros(7)
    Sx1 = np.zeros(7)
    Sxi0 = np.zeros(7)
    Sxi1 = np.zeros(7)
    
    for i in range(7):
        Sx0[i] = np.dot(p, I[i:i+7])
        Sx1[i] = np.dot(p, I[i+1:i+8])
        Sxi0[i] = np.dot(q, I[i:i+7])
        Sxi1[i] = np.dot(q, I[i+1:i+8])
        
    dMx0 = np.zeros(3)
    dMx1 = np.zeros(3)
    dMxi0 = np.zeros(3)
    dMxi1 = np.zeros(3)
    
    for m in range(3):
        Am = A_mat[:, m]
        dMx0[m] = 2.0 * np.dot(Am, Sx0)
        dMx1[m] = 2.0 * np.dot(Am, Sx1)
        dMxi0[m] = 2.0 * np.dot(Am, Sxi0)
        dMxi1[m] = 2.0 * np.dot(Am, Sxi1)
        
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    
    for m in range(3):
        dvx_m = (dMx1[m]*Mx0 - Mx1*dMx0[m]) / (Mx0*Mx0)
        dvxi_m = (dMxi1[m]*Mxi0 - Mxi1*dMxi0[m]) / (Mxi0*Mxi0)
        grad[m] = factor * (vxi*dvx_m + vx*dvxi_m)
        
    return bound, grad

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value: Known state of the art
    seed_params = np.array([-1.13, 0.04, 0.000308642], dtype=float)
    
    # Verify seed
    seed_val, _ = _compute_bound_and_grad(seed_params)
    if not np.isfinite(seed_val):
        # Fallback to a simple guess if seed is somehow invalid (should not happen)
        seed_params = np.array([0.0, 0.0, 0.0])
        seed_val, _ = _compute_bound_and_grad(seed_params)
        
    best_val = seed_val
    best_params = seed_params.copy()
    
    # Optimization strategy:
    # 1. Local refinement from the seed using L-BFGS-B.
    # 2. If it improves, keep it.
    # 3. Try a few very close perturbations to ensure we are at the local minimum.
    
    def _obj_and_grad(a):
        val, grad = _compute_bound_and_grad(a)
        return val, grad
        
    # Local refinement from seed
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_obj_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check, _ = _compute_bound_and_grad(candidate_a)
                if val_check < best_val and np.all(np.isfinite(candidate_a)):
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
            
    # Try Nelder-Mead for robust local search from the best found so far
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(lambda a: _compute_bound_and_grad(a)[0], best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check, _ = _compute_bound_and_grad(candidate_a)
                if val_check < best_val and np.all(np.isfinite(candidate_a)):
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
            
    # Final tight L-BFGS-B polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_obj_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check, _ = _compute_bound_and_grad(candidate_a)
                if val_check < best_val and np.all(np.isfinite(candidate_a)):
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
            
    # Safety check: if we ended up worse than the seed, return the seed
    if best_val > seed_val:
        best_params = seed_params.copy()
        
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
