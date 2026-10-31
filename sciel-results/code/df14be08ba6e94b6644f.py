import numpy as np
import time
from scipy.optimize import root

_DEADLINE = time.time() + 195

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Coefficients for Hermite polynomials in power basis x^{2k}
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# p_k is the coefficient of x^{2k} in P(x)
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
# a12 = -(a0 + 3*a4 + 105*a8) / 945

# Matrix A such that p = A * a, where a = [a0, a4, a8]
# p has 7 components (k=0..6)

H0_c = np.array([1.0, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3.0, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105.0, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945.0, -7560, 13860, -6930, 1155, -66, 1])

# a12 is linear in a: a12 = c12_0*a0 + c12_4*a4 + c12_8*a8
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

sign_vec = np.array([1.0, -1, 1, -1, 1, -1, 1])

def _get_p(a):
    return A_mat @ a

def _compute_variances(a):
    """
    Compute Vx and Vxi.
    Returns (Vx, Vxi, p, q)
    """
    p = _get_p(a)
    q = p * sign_vec
    
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
            idx0 = i + j
            idx1 = i + j + 1
            I0 = I[idx0]
            I1 = I[idx1]
            
            Mx0 += pi * p[j] * I0
            Mx1 += pi * p[j] * I1
            
            Mxi0 += qi * q[j] * I0
            Mxi1 += qi * q[j] * I1
            
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, 1e18, p, q
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, 1e18, p, q
        
    return vx, vxi, p, q

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound J = sqrt(Vx * Vxi) and its gradient.
    """
    vx, vxi, p, q = _compute_variances(a)
    
    if vx >= 1e17 or vxi >= 1e17:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation
    # We need dVx and dVxi
    # Sx[n, i] = sum_j p_j I[i+j+n]
    # Sxi[n, i] = sum_j q_j I[i+j+n]
    
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    # dM[n]_m = 2 * sum_i A[i,m] S[n, i]
    # Vx = Mx1 / Mx0
    # dVx_m = (dMx1_m * Mx0 - Mx1 * dMx0_m) / Mx0^2
    
    Mx0 = np.dot(p, I[0:7]) # Mx[0]
    Mx1 = np.dot(p, I[1:8]) # Mx[1]
    Mxi0 = np.dot(q, I[0:7])
    Mxi1 = np.dot(q, I[1:8])
    
    # Note: The loop in _compute_variances computed Mx0, Mx1 etc. 
    # Let's recompute them here to be safe and consistent with Sx calculation
    # Actually, Sx[0, i] = sum_j p_j I[i+j]. 
    # Mx0 = sum_i Sx[0, i] p_i ? No.
    # Mx0 = sum_{i,j} p_i p_j I[i+j]
    # Sx[0, i] = sum_j p_j I[i+j]
    # Mx0 = sum_i p_i Sx[0, i]
    
    Mx0_calc = np.dot(p, Sx[0])
    Mx1_calc = np.dot(p, Sx[1])
    Mxi0_calc = np.dot(q, Sxi[0])
    Mxi1_calc = np.dot(q, Sxi[1])
    
    if Mx0_calc <= 1e-15 or Mxi0_calc <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx_calc = Mx1_calc / Mx0_calc
    vxi_calc = Mxi1_calc / Mxi0_calc
    
    # dMx[n, m] = 2 * dot(A[:, m], Sx[n, :])
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx0_calc - Mx1_calc*dMx[0, m]) / (Mx0_calc*Mx0_calc)
        dvxi[m] = (dMxi[1, m]*Mxi0_calc - Mxi1_calc*dMxi[0, m]) / (Mxi0_calc*Mxi0_calc)
        
    # d(bound) = (1/(2*bound)) * (vxi*dvx + vx*dvxi)
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    for m in range(3):
        grad[m] = factor * (vxi_calc*dvx[m] + vx_calc*dvxi[m])
        
    return bound, grad

def _stationary_equations(a):
    """
    Returns the gradient of F = Vx * Vxi.
    We want to find roots of this system.
    """
    # We can reuse the gradient logic from _compute_bound_and_grad but for F
    # grad(F) = 2 * bound * grad(bound)
    # So we can just return 2 * bound * grad_bound
    bound, grad_bound = _compute_bound_and_grad(a)
    if bound >= 1e17:
        return np.array([1e18, 1e18, 1e18])
    return 2.0 * bound * grad_bound

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 10 # seconds
    
    # Generate diverse starting points for root finding
    np.random.seed(42)
    starts = [seed_params.copy()]
    
    # Random starts
    for _ in range(50):
        start = np.random.uniform(-5, 5, 3)
        starts.append(start)
        
    # Structured starts
    for a0 in [-2, -1, -0.5, 0.5, 1, 2]:
        for a4 in [-0.1, 0, 0.1]:
            for a8 in [-0.01, 0, 0.01]:
                starts.append(np.array([a0, a4, a8]))
                
    # Sort starts by initial objective value to prioritize promising regions
    def _init_score(s):
        v, _ = _compute_bound_and_grad(s)
        return v
    
    starts.sort(key=_init_score)
    
    # Run root finding from each start
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            # Use hybrid method for robustness
            res = root(_stationary_equations, start, method='hybrid', tol=1e-12)
            
            if res.success:
                # Verify it's a local minimum by checking the objective value
                # and optionally the Hessian (expensive, so just check value and gradient norm)
                val, grad = _compute_bound_and_grad(res.x)
                if np.isfinite(val) and val < best_val and np.linalg.norm(grad) < 1e-8:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish the best result with a local minimization if time permits
    # This helps if the root finder converged to a saddle point or if there's slight numerical error
    if time.time() < _DEADLINE - deadline_margin and best_val < seed_val:
        from scipy.optimize import minimize
        try:
            res_local = minimize(_compute_bound_and_grad[0], best_params, 
                                 jac=lambda x: _compute_bound_and_grad(x)[1], 
                                 method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.isfinite(res_local.fun) and res_local.fun < best_val:
                best_val = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass
            
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
