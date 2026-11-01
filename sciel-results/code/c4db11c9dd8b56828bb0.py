"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Deadline guard
_START_TIME = time.time()
DEADLINE = 210.0
SAFETY_MARGIN = 15.0

def _check_deadline():
    return (time.time() - _START_TIME) < (DEADLINE - SAFETY_MARGIN)

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n-1)!! * sqrt(pi) / 2^n
# Using recurrence: I_0 = sqrt(pi), I_n = (2n-1)/2 * I_{n-1}
_I_max = 25  # We need up to index 24 (x^24) for m4 of degree 12 poly
_I = np.zeros(_I_max + 1)
_I[0] = np.sqrt(np.pi)
for n in range(1, _I_max + 1):
    _I[n] = (2.0 * n - 1.0) / 2.0 * _I[n - 1]

def _get_poly_coeffs(a0, a4, a8):
    """
    Return coefficients p[0..6] for P(x) = sum_{k=0}^6 p[k] x^{2k}
    where P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    and a12 = -(a0 + 3*a4 + 105*a8) / 945.0
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    p = np.zeros(7)
    # p0 (x^0)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    # p1 (x^2)
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    # p2 (x^4)
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    # p3 (x^6)
    p[3] = -28.0 * a8 - 6930.0 * a12
    # p4 (x^8)
    p[4] = a8 + 1155.0 * a12
    # p5 (x^10)
    p[5] = -66.0 * a12
    # p6 (x^12)
    p[6] = a12
    
    return p

# Precompute moment matrices for vectorized computation
# M_k[i,j] = I[k+i+j] for i,j in 0..6
_M0 = np.zeros((7, 7))
_M1 = np.zeros((7, 7))
_M2 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        _M0[i, j] = _I[i + j]
        _M1[i, j] = _I[1 + i + j]
        _M2[i, j] = _I[2 + i + j]

# Signs for Fourier side: s[k] = (-1)^k
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

def _compute_moments(p):
    """
    Compute moments m_k = integral x^{2k} |P(x)|^2 e^{-x^2} dx for k=0,1,2
    and n_k for the Fourier side, using precomputed matrices.
    """
    # m_k = p^T M_k p
    m0 = p @ _M0 @ p
    m1 = p @ _M1 @ p
    m2 = p @ _M2 @ p
    
    # Fourier side: q = p * signs
    q = p * _signs
    
    # n_k = 2pi * q^T M_k q
    n0 = 2.0 * np.pi * (q @ _M0 @ q)
    n1 = 2.0 * np.pi * (q @ _M1 @ q)
    n2 = 2.0 * np.pi * (q @ _M2 @ q)
    
    return np.array([m0, m1, m2]), np.array([n0, n1, n2])

# Precompute derivative matrices for each parameter
# J[c, k] = dp_k/da_c (direct part, not including a12 chain)
_J_direct = np.array([
    [1.0, 0.0, 0.0],   # p0 wrt a0,a4,a8
    [0.0, -6.0, 0.0],  # p1
    [0.0, 1.0, 0.0],   # p2
    [0.0, 0.0, -28.0], # p3
    [0.0, 0.0, 1.0],   # p4
    [0.0, 0.0, 0.0],   # p5
    [0.0, 0.0, 0.0]    # p6
])

_a12_coeff = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# Full Jacobian: dp/da_c for c=0,1,2
_J_full = np.zeros((7, 3))
for c in range(3):
    _J_full[:, c] = _J_direct[:, c] + _a12_coeff * _da12[c]

def _objective(params):
    """
    Compute the uncertainty product bound with vectorized gradient.
    Returns (value, gradient).
    """
    a0, a4, a8 = params[0], params[1], params[2]
    p = _get_poly_coeffs(a0, a4, a8)
    m, n = _compute_moments(p)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Vectorized gradient computation
    # dm_k/da_c = 2 * dp_c^T M_k p
    # dn_k/da_c = 2pi * 2 * dq_c^T M_k q = 4pi * (dp_c * signs)^T M_k q
    
    q = p * _signs
    grad = np.zeros(3)
    
    for c in range(3):
        dp = _J_full[:, c]
        dq = dp * _signs
        
        dm0 = 2.0 * (dp @ _M0 @ p)
        dm1 = 2.0 * (dp @ _M1 @ p)
        dm2 = 2.0 * (dp @ _M2 @ p)
        
        dn0 = 4.0 * np.pi * (dq @ _M0 @ q)
        dn1 = 4.0 * np.pi * (dq @ _M1 @ q)
        dn2 = 4.0 * np.pi * (dq @ _M2 @ q)
        
        dvm2 = (dm1 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn1 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
        
    return bound, grad

def _objective_val(params):
    val, _ = _objective(params)
    return val

def construct():
    if not _check_deadline():
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = None
    best_score = np.inf
    
    # Seed point
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # 1. Local optimization from seed with analytic gradient
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
    except Exception:
        pass
    
    if not _check_deadline():
        if best_result is not None:
            return best_result
        return x0
    
    # 2. Global search with Differential Evolution (reduced to save time for polish)
    bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
    try:
        de_result = differential_evolution(_objective_val, bounds, 
                                           maxiter=200, popsize=25,
                                           seed=42, tol=1e-10,
                                           polish=False, 
                                           init='sobol')
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
    except Exception:
        pass
    
    if not _check_deadline():
        if best_result is not None:
            return best_result
        return x0
        
    # 3. Multi-start local refinement with structured starts
    rng = np.random.default_rng(42)
    base = best_result if best_result is not None else x0
    
    # Structured starts: perturbations at multiple scales
    starts = [x0, base.copy()]
    
    # Radial perturbations at multiple scales
    for scale in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for _ in range(5):
            direction = rng.normal(0, 1, 3)
            direction /= (np.linalg.norm(direction) + 1e-15)
            starts.append(base + scale * direction)
    
    # Axis-aligned perturbations
    for dim in range(3):
        for delta in [-0.1, -0.05, 0.05, 0.1]:
            pert = base.copy()
            pert[dim] += delta
            starts.append(pert)
    
    # Scaled seed points
    for scale in [0.5, 0.75, 1.25, 1.5, 2.0]:
        starts.append(x0 * scale)
    
    for start in starts:
        if not _check_deadline():
            break
        try:
            res = minimize(_objective, start, method='L-BFGS-B', jac=True,
                          options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    # 4. High-precision Newton-style polish with tight tolerances
    if best_result is not None and _check_deadline():
        # First, a very tight L-BFGS-B
        try:
            tight_result = minimize(_objective, best_result, method='L-BFGS-B', jac=True,
                                   options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-18,
                                            'eps': 1e-12})
            if tight_result.fun < best_score and np.isfinite(tight_result.fun):
                best_score = tight_result.fun
                best_result = tight_result.x
        except Exception:
            pass
        
        # Nelder-Mead for potential escape from flat regions
        if _check_deadline():
            try:
                nm_result = minimize(_objective_val, best_result, method='Nelder-Mead',
                                   options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-16,
                                            'adaptive': True, 'initial_simplex': 1e-8})
                if nm_result.fun < best_score and np.isfinite(nm_result.fun):
                    best_score = nm_result.fun
                    best_result = nm_result.x
            except Exception:
                pass
        
        # Final L-BFGS-B from Nelder-Mead result
        if _check_deadline():
            try:
                final_result = minimize(_objective, best_result, method='L-BFGS-B', jac=True,
                                       options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-18,
                                                'eps': 1e-14})
                if final_result.fun < best_score and np.isfinite(final_result.fun):
                    best_score = final_result.fun
                    best_result = final_result.x
            except Exception:
                pass
    
    if best_result is not None:
        return best_result
    return x0

# EVOLVE-BLOCK-END
