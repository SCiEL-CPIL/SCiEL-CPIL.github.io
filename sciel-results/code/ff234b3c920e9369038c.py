"""Analytic uncertainty bound with exact gradient + multi-method optimization."""
import numpy as np
import time
import math
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 195  # 15s safety margin before 210s kill

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    sp = math.sqrt(math.pi)
    I = np.zeros(14)
    for n in range(14):
        # (2n)! / (4^n n!) * sqrt(pi)
        num = 1.0
        for m in range(1, 2*n + 1):
            num *= m
        den = (4.0 ** n)
        for m in range(1, n + 1):
            den *= m
        I[n] = num / den * sp
    return I

_I_TABLE = _precompute_I()

# Precompute the coefficient matrix for p in terms of (a0, a4, a8, a12)
# p0 = 1*a0 + 3*a4 + 105*a8 + 945*a12
# p1 = 0*a0 + (-6)*a4 + (-420)*a8 + (-7560)*a12
# p2 = 0*a0 + 1*a4 + 210*a8 + 13860*a12
# p3 = 0*a0 + 0*a4 + (-28)*a8 + (-6930)*a12
# p4 = 0*a0 + 0*a4 + 1*a8 + 1155*a12
# p5 = 0*a0 + 0*a4 + 0*a8 + (-66)*a12
# p6 = 0*a0 + 0*a4 + 0*a8 + 1*a12

# Constraint: g(0) = 0 => p0 = 0 => a0 + 3*a4 + 105*a8 + 945*a12 = 0
# So a12 = -(a0 + 3*a4 + 105*a8) / 945

# dp_k/da_c for c in {0, 4, 8}, accounting for a12 dependency
# da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945

# For each coefficient c, the derivative of p_k with respect to a_c:
# dp_k/da_c = (direct coefficient of a_c in p_k) + (coefficient of a12 in p_k) * (da12/da_c)

# Direct coefficients of a0, a4, a8 in p_k:
# p0: [1, 3, 105]
# p1: [0, -6, -420]
# p2: [0, 1, 210]
# p3: [0, 0, -28]
# p4: [0, 0, 1]
# p5: [0, 0, 0]
# p6: [0, 0, 0]

# Coefficients of a12 in p_k:
# p0: 945, p1: -7560, p2: 13860, p3: -6930, p4: 1155, p5: -66, p6: 1

# da12/da_c: [-1/945, -3/945, -105/945]

# So dp_k/da_c = direct[c] + a12_coeff * da12_dc[c]
# = direct[c] + a12_coeff * [-1/945, -3/945, -105/945][c]

# Let's precompute dp_k_dc for each k and c:
# dp_k_dc[c] = direct_coeff[k][c] + a12_coeff[k] * da12_dc[c]

# direct_coeff:
_direct = np.array([
    [1.0, 3.0, 105.0],   # p0
    [0.0, -6.0, -420.0], # p1
    [0.0, 1.0, 210.0],   # p2
    [0.0, 0.0, -28.0],   # p3
    [0.0, 0.0, 1.0],     # p4
    [0.0, 0.0, 0.0],     # p5
    [0.0, 0.0, 0.0],     # p6
])

_a12_coeff = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
_da12_dc = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# dp_k_dc[k][c] = _direct[k][c] + _a12_coeff[k] * _da12_dc[c]
_dp_dc = np.zeros((7, 3))
for k in range(7):
    for c in range(3):
        _dp_dc[k][c] = _direct[k][c] + _a12_coeff[k] * _da12_dc[c]


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I_TABLE
    
    # m_k = integral x^{2k} |P|^2 e^{-x^2} dx = sum_{i,j} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pp = pi * p[j]
            if pp == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pp * I[idx]
    
    # For Fourier transform: Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qq = qi * q[j]
            if qq == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0*np.pi * qq * I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound) or bound <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    # Compute gradient analytically
    grad = np.zeros(3)
    for c in range(3):
        # dp = dp_dc[:, c]
        dp = _dp_dc[:, c]
        
        # dm_k/da_c = 2*sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += p[j] * I[idx]
                dm[k] += 2.0 * dpi * s
        
        # dq_k/da_c = dp_k * (-1)^k
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0)**k)
        
        # dn_k/da_c = 2*2*pi*sum_i dq_i * sum_j q_j I[k+i+j]
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += q[j] * I[idx]
                dn[k] += 4.0 * np.pi * dqi * s
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    """
    Returns three coefficients [a0, a4, a8] that minimize the uncertainty product.
    The a12 coefficient is determined by the constraint g(0) = 0.
    """
    # Multiple starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([0.5, -0.02, 0.0001]),
        np.array([-0.8, 0.06, 0.0006]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(30):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with COBYLA if time permits
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='COBYLA',
                          options={'maxiter': 10000, 'rhobeg': 1e-8, 'tol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([-1.13, 0.04, 0.000308642])
    
    return best_params
