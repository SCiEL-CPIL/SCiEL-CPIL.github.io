"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
_I_TABLE = np.zeros(14)
def _init_I():
    sp = np.sqrt(np.pi)
    _I_TABLE[0] = sp
    for n in range(1, 14):
        _I_TABLE[n] = _I_TABLE[n-1] * (2*n - 1) / (4.0 * n)
_init_I()

# Precompute sign pattern for Fourier: q_k = p_k * (-1)^k
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Matrix M[k, i, j] = I[k+i+j] for k=0,1,2; i,j=0..6
_M = np.zeros((3, 7, 7))
for k in range(3):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                _M[k, i, j] = _I_TABLE[idx]

# dp/da_c matrix: dp_dc[c, k] = d(p_k)/d(a_c)
# p0 = a0+3a4+105a8+945a12
# p1 = -6a4-420a8-7560a12
# p2 = a4+210a8+13860a12
# p3 = -28a8-6930a12
# p4 = a8+1155a12
# p5 = -66a12
# p6 = a12
# da12/da_c = [-1/945, -3/945, -105/945]
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
_dp_dc = np.zeros((3, 7))
_dp_dc[0, 0] = 1.0 + 945.0 * _da12[0]
_dp_dc[1, 0] = 3.0 + 945.0 * _da12[1]
_dp_dc[2, 0] = 105.0 + 945.0 * _da12[2]
_dp_dc[0, 1] = 0.0 - 7560.0 * _da12[0]
_dp_dc[1, 1] = -6.0 - 7560.0 * _da12[1]
_dp_dc[2, 1] = -420.0 - 7560.0 * _da12[2]
_dp_dc[0, 2] = 0.0 + 13860.0 * _da12[0]
_dp_dc[1, 2] = 1.0 + 13860.0 * _da12[1]
_dp_dc[2, 2] = 210.0 + 13860.0 * _da12[2]
_dp_dc[0, 3] = 0.0 - 6930.0 * _da12[0]
_dp_dc[1, 3] = 0.0 - 6930.0 * _da12[1]
_dp_dc[2, 3] = -28.0 - 6930.0 * _da12[2]
_dp_dc[0, 4] = 0.0 + 1155.0 * _da12[0]
_dp_dc[1, 4] = 0.0 + 1155.0 * _da12[1]
_dp_dc[2, 4] = 1.0 + 1155.0 * _da12[2]
_dp_dc[0, 5] = 0.0 - 66.0 * _da12[0]
_dp_dc[1, 5] = 0.0 - 66.0 * _da12[1]
_dp_dc[2, 5] = 0.0 - 66.0 * _da12[2]
_dp_dc[0, 6] = 0.0 + 1.0 * _da12[0]
_dp_dc[1, 6] = 0.0 + 1.0 * _da12[1]
_dp_dc[2, 6] = 0.0 + 1.0 * _da12[2]


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.array([
        a0 + 3.0*a4 + 105.0*a8 + 945.0*a12,
        -6.0*a4 - 420.0*a8 - 7560.0*a12,
        a4 + 210.0*a8 + 13860.0*a12,
        -28.0*a8 - 6930.0*a12,
        a8 + 1155.0*a12,
        -66.0*a12,
        a12
    ])
    
    # m_k = sum_{i,j} p_i p_j I[k+i+j] = p^T M[k] p
    m0 = np.dot(p, np.dot(_M[0], p))
    m2 = np.dot(p, np.dot(_M[1], p))
    
    # q_k = p_k * (-1)^k
    q = p * _SIGNS
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    two_pi = 2.0 * np.pi
    n0 = two_pi * np.dot(q, np.dot(_M[0], q))
    n2 = two_pi * np.dot(q, np.dot(_M[1], q))
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient: d(bound)/d(a_c) = (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c)) / (2*bound)
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # dm_k/da_c = 2 * dp_dc[c] @ M[k] @ p
    # Similarly for n with q
    
    grad = np.zeros(3)
    inv_m0sq = 1.0 / (m0 * m0)
    inv_n0sq = 1.0 / (n0 * n0)
    
    for c in range(3):
        dp = _dp_dc[c]
        
        # dm0/da_c = 2 * dp @ M[0] @ p
        dm0 = 2.0 * np.dot(dp, np.dot(_M[0], p))
        dm2 = 2.0 * np.dot(dp, np.dot(_M[1], p))
        
        # dq = dp * signs
        dq = dp * _SIGNS
        dn0 = 2.0 * two_pi * np.dot(dq, np.dot(_M[0], q))
        dn2 = 2.0 * two_pi * np.dot(dq, np.dot(_M[1], q))
        
        dvm2 = (dm2 * m0 - m2 * dm0) * inv_m0sq
        dvn2 = (dn2 * n0 - n2 * dn0) * inv_n0sq
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def _log_param_to_linear(log_params):
    """Convert log-magnitude parameters to linear coefficients."""
    signs = np.array([-1.0, 1.0, 1.0])  # a0<0, a4>0, a8>0 expected
    mags = np.exp(log_params)
    return signs * mags


def _linear_to_log_param(params):
    """Convert linear coefficients to log-magnitude parameters."""
    signs = np.array([-1.0, 1.0, 1.0])
    mags = np.abs(params)
    # Avoid log(0)
    mags = np.maximum(mags, 1e-30)
    return np.log(mags)


def _log_objective_grad(log_params):
    """Objective and gradient in log-parameter space."""
    params = _log_param_to_linear(log_params)
    val, grad = _bound_and_grad(params)
    if not np.isfinite(val):
        return 1e18, np.zeros(3)
    # Chain rule: d(val)/d(log_params) = d(val)/d(params) * d(params)/d(log_params)
    # d(params)/d(log_params) = signs * exp(log_params) = params (element-wise)
    # But sign is fixed, so d(params_i)/d(log_params_i) = params_i
    log_grad = grad * params
    return val, log_grad


def construct():
    # Multiple starting points in linear space
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0004]),
        np.array([-1.5, 0.06, 0.0006]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(30):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from multiple starts in linear space
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Log-magnitude optimization from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            log_start = _linear_to_log_param(best_params)
            res = minimize(_log_objective_grad, log_start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = _log_param_to_linear(res.x)
        except Exception:
            pass
    
    # Phase 3: Polish with Nelder-Mead
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
