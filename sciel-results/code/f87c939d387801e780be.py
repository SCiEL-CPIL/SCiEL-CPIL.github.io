"""
Uncertainty inequality: tune Fourier-invariant Hermite-Gaussian test function.
Strategy: reparameterize onto the unit sphere to remove scale, then use
stochastic local search with adaptive step sizes + high-precision L-BFGS-B polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..12
def _precompute_integrals():
    I = np.zeros(13)
    sp = math.sqrt(math.pi)
    for n in range(13):
        num = math.factorial(2 * n)
        den = (4.0 ** n) * math.factorial(n)
        I[n] = (num / den) * sp
    return I

_I_TABLE = _precompute_integrals()

# Precompute the matrix M such that m_k = p^T M_k p
# where p = [p0..p6] are coefficients of P(x) = sum p_i x^{2i}
# m_k = sum_{i,j} p_i p_j I[k+i+j]
# So M_k[i,j] = I[k+i+j] if k+i+j <= 12 else 0

def _precompute_moment_matrices():
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
    return M0, M2

_M0, _M2 = _precompute_moment_matrices()

# Mapping from (c0, c4, c8) to polynomial coefficients p
# c12 = -(c0 + 3*c4 + 105*c8) / 945
# p[0] = c0 + 3*c4 + 105*c8 + 945*c12 = 0  (by construction!)
# p[1] = -6*c4 - 420*c8 - 7560*c12
# p[2] = c4 + 210*c8 + 13860*c12
# p[3] = -28*c8 - 6930*c12
# p[4] = c8 + 1155*c12
# p[5] = -66*c12
# p[6] = c12

# So p is linear in (c0, c4, c8). Let's build the 7x3 matrix A such that p = A @ c
# where c = [c0, c4, c8]
# c12 = (-1/945)*c0 + (-3/945)*c4 + (-105/945)*c8
# Let dc12 = [-1/945, -3/945, -105/945]

def _build_A_matrix():
    A = np.zeros((7, 3))
    dc12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # p[0] = c0 + 3*c4 + 105*c8 + 945*c12 = 0 (always, by constraint)
    A[0, :] = np.array([1.0, 3.0, 105.0]) + 945.0 * dc12  # This should be [0,0,0]
    
    # p[1] = -6*c4 - 420*c8 - 7560*c12
    A[1, :] = np.array([0.0, -6.0, -420.0]) - 7560.0 * dc12
    
    # p[2] = c4 + 210*c8 + 13860*c12
    A[2, :] = np.array([0.0, 1.0, 210.0]) + 13860.0 * dc12
    
    # p[3] = -28*c8 - 6930*c12
    A[3, :] = np.array([0.0, 0.0, -28.0]) - 6930.0 * dc12
    
    # p[4] = c8 + 1155*c12
    A[4, :] = np.array([0.0, 0.0, 1.0]) + 1155.0 * dc12
    
    # p[5] = -66*c12
    A[5, :] = -66.0 * dc12
    
    # p[6] = c12
    A[6, :] = dc12
    
    return A

_A = _build_A_matrix()

def _bound_from_c(c):
    """Compute the uncertainty bound from coefficients c = [c0, c4, c8]."""
    p = _A @ c
    m0 = p @ (_M0 @ p)
    m2 = p @ (_M2 @ p)
    if m0 <= 0 or not np.isfinite(m0):
        return 1e18
    var_x = m2 / m0
    if var_x <= 0 or not np.isfinite(var_x):
        return 1e18
    return var_x  # bound = sqrt(var_x * var_xi) = sqrt(var_x^2) = var_x

def _bound_grad_from_c(c):
    """Compute bound and gradient with respect to c."""
    p = _A @ c
    m0 = p @ (_M0 @ p)
    m2 = p @ (_M2 @ p)
    if m0 <= 0 or not np.isfinite(m0):
        return 1e18, np.array([0.0, 0.0, 0.0])
    
    var_x = m2 / m0
    
    # Gradient: dvar_x/dc = (dm2/m0 - m2*dm0/m0^2)
    # dm0/dc = 2 * (A^T M0 p)
    # dm2/dc = 2 * (A^T M2 p)
    dm0_dc = 2.0 * (_A.T @ (_M0 @ p))
    dm2_dc = 2.0 * (_A.T @ (_M2 @ p))
    
    dvar_x = (dm2_dc * m0 - m2 * dm0_dc) / (m0 * m0)
    
    if not np.isfinite(dvar_x).all():
        return var_x, np.zeros(3)
    
    return var_x, dvar_x

def _objective(c):
    return _bound_from_c(c)

def _objective_grad(c):
    val, grad = _bound_grad_from_c(c)
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 170  # Safety margin before 180s limit
    
    best_c = np.array([-1.13, 0.04, 0.000308642])
    best_score = _bound_from_c(best_c)
    
    # --- Phase 1: Sphere-based stochastic local search ---
    # Normalize to unit sphere to remove scale invariance
    # The bound is homogeneous of degree 0 in c (scaling c doesn't change the bound),
    # so we can restrict to ||c|| = 1.
    
    def _project_to_sphere(c):
        norm = np.linalg.norm(c)
        if norm < 1e-15:
            return np.array([1.0, 0.0, 0.0])
        return c / norm
    
    # Start from best_c projected to sphere
    c_sphere = _project_to_sphere(best_c)
    
    # Adaptive step size stochastic search on the sphere
    rng = np.random.RandomState(42)
    step_size = 0.1
    best_sphere_score = _bound_from_c(c_sphere)
    best_c = c_sphere.copy()
    best_score = best_sphere_score
    
    # Generate diverse starting points on the sphere
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 60:
            break
        # Random point on sphere
        v = rng.randn(3)
        c_try = _project_to_sphere(v)
        # Also try scaled versions (the bound is scale-invariant, but let's be safe)
        score_try = _bound_from_c(c_try)
        if score_try < best_score:
            best_score = score_try
            best_c = c_try.copy()
    
    # Local stochastic search from best point
    c_current = best_c.copy()
    step_size = 0.05
    for it in range(500):
        if time.time() > deadline - 50:
            break
        # Generate random perturbation
        delta = rng.randn(3) * step_size
        c_new = c_current + delta
        c_new = _project_to_sphere(c_new)
        
        score_new = _bound_from_c(c_new)
        if score_new < best_score:
            best_score = score_new
            best_c = c_new.copy()
            c_current = c_new.copy()
            step_size = min(step_size * 1.1, 0.5)
        else:
            # Accept with some probability (simulated annealing-like)
            if rng.random() < 0.1:
                c_current = c_new.copy()
            step_size = max(step_size * 0.98, 0.001)
    
    # --- Phase 2: High-precision L-BFGS-B polish ---
    if time.time() < deadline - 30:
        try:
            res = minimize(_objective_grad, best_c, jac=True, method='L-BFGS-B',
                          options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # --- Phase 3: Additional restarts with L-BFGS-B ---
    if time.time() < deadline - 10:
        for i in range(10):
            if time.time() > deadline - 5:
                break
            v = rng.randn(3)
            c_start = _project_to_sphere(v)
            # Scale to a reasonable magnitude
            c_start = c_start * 1.0
            try:
                res = minimize(_objective_grad, c_start, jac=True, method='L-BFGS-B',
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-13})
                if res.fun < best_score:
                    best_score = res.fun
                    best_c = res.x.copy()
            except Exception:
                pass
    
    # --- Phase 4: Final high-precision polish ---
    if time.time() < deadline - 2:
        try:
            res = minimize(_objective_grad, best_c, jac=True, method='L-BFGS-B',
                          options={'maxiter': 20000, 'ftol': 1e-20, 'gtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    return best_c

# EVOLVE-BLOCK-END
