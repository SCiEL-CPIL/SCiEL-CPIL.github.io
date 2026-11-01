import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

# Precompute Hermite polynomial coefficients for H_0, H_4, H_8, H_12
# Using physicists' convention: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 3072x^6 + 13440x^4 - 13440x^2 + 1680
# H_12 = 4096x^12 - 98304x^10 + 1290240x^8 - 6451200x^6 + 12902400x^4 - 6451200x^2 + 665280

# Coefficients for x^(2k) in each Hermite polynomial
_H_COEFFS = np.array([
    [1.0, 0, 0, 0, 0, 0, 0],           # H_0
    [12.0, -48.0, 16.0, 0, 0, 0, 0],   # H_4
    [1680.0, -13440.0, 13440.0, -3072.0, 256.0, 0, 0],  # H_8
    [665280.0, -6451200.0, 12902400.0, -6451200.0, 1290240.0, -98304.0, 4096.0]  # H_12
])

# H_n(0) values for the constraint g(0) = 0
_H_AT_ZERO = np.array([1.0, 12.0, 1680.0, 665280.0])

# Precompute I_n = ∫ x^(2n) e^{-x^2} dx = (2n)!/(4^n n!) * sqrt(pi) for n=0..14
def _compute_I_table():
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            # (2n)! / (4^n * n!) * sqrt(pi)
            log_num = sum(math.log(k) for k in range(1, 2*n+1))
            log_den = n * math.log(4) + sum(math.log(k) for k in range(1, n+1))
            I[n] = math.exp(log_num - log_den) * np.sqrt(np.pi)
    return I

_I_TABLE = _compute_I_table()

# Matrix M where M[k, i, j] = I_{k+i+j} for k=0,1; i,j=0..6
# This allows vectorized computation of moments
_M_SPATIAL = np.zeros((2, 7, 7))
_M_FREQ = np.zeros((2, 7, 7))
for k in range(2):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 15:
                _M_SPATIAL[k, i, j] = _I_TABLE[idx]
                _M_FREQ[k, i, j] = 2.0 * np.pi * _I_TABLE[idx]

# Signs for frequency domain: q_k = p_k * (-1)^k
_SIGNS = np.array([(-1.0)**k for k in range(7)])

def _compute_bound_and_grad(params):
    """
    Compute the uncertainty bound sqrt(<x^2><xi^2>) and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 is determined by g(0)=0: a12 = -(a0 + 12*a4 + 1680*a8) / 665280
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Compute a12 from constraint g(0) = 0
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    
    # Compute polynomial coefficients p[k] for x^(2k)
    p = _H_COEFFS @ coeffs  # shape (7,)
    
    # Compute spatial moments m_k = ∫ x^(2k) |P(x)|^2 e^{-x^2} dx
    # m_k = sum_i sum_j p_i p_j I_{k+i+j}
    # Using matrix form: m_k = p^T M_k p
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    
    # Compute frequency moments
    # Q(xi) = P(-i*xi), so q_k = p_k * (-1)^k
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient
    # da12/da_c = -H_c(0) / H_12(0)
    da12_vec = -_H_AT_ZERO[:3] / _H_AT_ZERO[3]
    
    grad = np.zeros(3)
    for c_idx in range(3):
        # dp/da_c = H_COEFFS[c_idx] + da12/da_c * H_COEFFS[3]
        dp = _H_COEFFS[c_idx] + da12_vec[c_idx] * _H_COEFFS[3]
        
        # dm_k/da_c = 2 * dp^T M_k p
        dm = 2.0 * np.array([dp @ _M_SPATIAL[k] @ p for k in range(2)])
        
        # dq/da_c = dp * signs
        dq = dp * _SIGNS
        
        # dn_k/da_c = 2 * dq^T M_FREQ_k q
        dn = 2.0 * np.array([dq @ _M_FREQ[k] @ q for k in range(2)])
        
        # d(vf)/da_c = (dm2*m0 - m2*dm0) / m0^2
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        # d(vF)/da_c = (dn2*n0 - n2*dn0) / n0^2
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c_idx] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def _ratio_objective(uv):
    """
    Objective in ratio space: u = a4/a0, v = a8/a0.
    Scale invariance means we can fix a0 = 1 and compute a4 = u, a8 = v.
    The bound is homogeneous of degree 0 in (a0, a4, a8), so fixing a0=1 is valid.
    """
    u, v = uv[0], uv[1]
    a0 = 1.0
    a4 = u
    a8 = v
    val, _ = _compute_bound_and_grad(np.array([a0, a4, a8]))
    return val

def _ratio_grad(uv):
    """
    Gradient of the ratio objective.
    d/d(a0) is not needed since we fix a0=1.
    d/d(a4) = grad[1], d/d(a8) = grad[2]
    """
    u, v = uv[0], uv[1]
    a0 = 1.0
    a4 = u
    a8 = v
    val, grad = _compute_bound_and_grad(np.array([a0, a4, a8]))
    # The gradient w.r.t. (u, v) is just (grad[1], grad[2]) since a4=u, a8=v
    return val, np.array([grad[1], grad[2]])

def construct():
    """
    Global optimization via differential_evolution in ratio space,
    followed by local refinement with L-BFGS-B and Nelder-Mead.
    Returns three coefficients (a0, a4, a8) for H0, H4, H8.
    """
    best_val = np.inf
    best_params = None
    
    def _try_optimize(start, method='L-BFGS-B', maxiter=5000, ftol=1e-16, gtol=1e-14):
        """Helper to try optimization from a starting point."""
        nonlocal best_val, best_params
        try:
            if method == 'L-BFGS-B':
                res = minimize(_objective_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            elif method == 'Nelder-Mead':
                res = minimize(_objective, start, method='Nelder-Mead',
                              options={'maxiter': maxiter, 'xatol': 1e-16, 'fatol': 1e-16})
            else:
                return
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 1: Global search using differential_evolution in ratio space
    # u = a4/a0, v = a8/a0
    # Based on known good point: a0=-1.13, a4=0.04, a8=0.000308642
    # u = 0.04/-1.13 ≈ -0.0354, v = 0.000308642/-1.13 ≈ -0.000273
    
    # Define bounds for u and v
    # u range: roughly -0.5 to 0.5
    # v range: roughly -0.01 to 0.01
    bounds_ratio = [(-0.5, 0.5), (-0.01, 0.01)]
    
    if time.time() < _DEADLINE - 60:
        try:
            de_result = differential_evolution(
                _ratio_objective,
                bounds_ratio,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-12,
                polish=True,
                init='sobol',
                updating='immediate'
            )
            if de_result.fun < best_val and np.isfinite(de_result.fun):
                u_best, v_best = de_result.x
                a0_best = 1.0
                a4_best = u_best
                a8_best = v_best
                best_val = de_result.fun
                best_params = np.array([a0_best, a4_best, a8_best])
        except Exception:
            pass
    
    # Phase 2: Local refinement from multiple starting points
    # Add known good point and its neighborhood
    known_good = np.array([-1.13, 0.04, 0.000308642])
    
    # Add the DE result if found
    if best_params is not None:
        starts = [best_params.copy()]
    else:
        starts = []
    
    # Add known good point
    starts.append(known_good)
    
    # Add neighborhood around known good
    for da0 in np.linspace(-0.5, 0.5, 7):
        for da4 in np.linspace(-0.05, 0.05, 7):
            for da8 in np.linspace(-0.0005, 0.0005, 5):
                starts.append(known_good + np.array([da0, da4, da8]))
    
    # Add some random starts
    rng = np.random.default_rng(12345)
    for scale in [0.5, 1.0, 2.0, 5.0]:
        for _ in range(10):
            starts.append(rng.normal(0, scale, 3))
    
    # Try optimization from each start
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        _try_optimize(start, method='L-BFGS-B', maxiter=5000, ftol=1e-16, gtol=1e-14)
    
    # Phase 3: Polish with Nelder-Mead and tighter L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 15:
        # Nelder-Mead polish
        _try_optimize(best_params, method='Nelder-Mead', maxiter=20000, ftol=1e-16, gtol=1e-14)
        
        # L-BFGS-B with tighter tolerances
        if time.time() < _DEADLINE - 10:
            _try_optimize(best_params, method='L-BFGS-B', maxiter=10000, ftol=1e-18, gtol=1e-16)
    
    # Phase 4: Fine grid around best point if time remains
    if best_params is not None and time.time() < _DEADLINE - 5:
        bp = best_params
        fine_starts = []
        for da0 in np.linspace(-0.1, 0.1, 7):
            for da4 in np.linspace(-0.01, 0.01, 7):
                for da8 in np.linspace(-0.0001, 0.0001, 5):
                    fine_starts.append(bp + np.array([da0, da4, da8]))
        
        for start in fine_starts:
            if time.time() > _DEADLINE - 3:
                break
            _try_optimize(start, method='L-BFGS-B', maxiter=2000, ftol=1e-16, gtol=1e-14)
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    return best_params
