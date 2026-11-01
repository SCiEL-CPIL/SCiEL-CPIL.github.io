"""Analytic uncertainty bound with corrected constraint and robust optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200

# Precompute Gaussian moments I_n = integral x^(2n) e^(-x^2) dx
# Note: The test function is g(x) = P(x) e^{-x^2/2}.
# The Fourier transform of e^{-x^2/2} is sqrt(2*pi) e^{-k^2/2}.
# The integrals for variance involve e^{-x^2} because |g(x)|^2 = P(x)^2 e^{-x^2}.
# So I_n = integral x^(2n) e^{-x^2} dx is correct.
_I_CACHE = None
def _get_I():
    global _I_CACHE
    if _I_CACHE is None:
        _I_CACHE = np.zeros(15)
        _I_CACHE[0] = np.sqrt(np.pi)
        for n in range(1, 15):
            _I_CACHE[n] = _I_CACHE[n-1] * (2*n - 1) / 2.0
    return _I_CACHE

# Precompute matrices M_k for m_k = p^T M_k p
# (M_k)_{ij} = I_{k+i+j}
_M_CACHES = None
def _get_M():
    global _M_CACHES
    if _M_CACHES is None:
        I = _get_I()
        _M_CACHES = []
        for k in range(3):
            M = np.zeros((7, 7))
            for i in range(7):
                for j in range(7):
                    M[i, j] = I[k + i + j]
            _M_CACHES.append(M)
    return _M_CACHES

# Corrected values for H_n(0) for Probabilists' Hermite polynomials
# H_0(0) = 1
# H_4(0) = 3
# H_8(0) = 105
# H_12(0) = 10395
_H0_0 = 1.0
_H4_0 = 3.0
_H8_0 = 105.0
_H12_0 = 10395.0

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Corrected constraint: P(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # a12 = -(a0*1 + a4*3 + a8*105) / 10395
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 10395
    
    p = np.zeros(7)
    # x^0
    p[0] = a0 * 1.0 + a4 * 3.0 + a8 * 105.0 + a12 * 10395.0
    # x^2
    p[1] = a0 * 0.0 + a4 * (-6.0) + a8 * (-420.0) + a12 * (-7560.0)
    # x^4
    p[2] = a0 * 0.0 + a4 * 1.0 + a8 * 210.0 + a12 * 13860.0
    # x^6
    p[3] = a0 * 0.0 + a4 * 0.0 + a8 * (-28.0) + a12 * (-6930.0)
    # x^8
    p[4] = a0 * 0.0 + a4 * 0.0 + a8 * 1.0 + a12 * 1155.0
    # x^10
    p[5] = a0 * 0.0 + a4 * 0.0 + a8 * 0.0 + a12 * (-66.0)
    # x^12
    p[6] = a0 * 0.0 + a4 * 0.0 + a8 * 0.0 + a12 * 1.0
    
    # q_k = p_k * (-1)^k
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    M = _get_M()
    M0, M1, M2 = M[0], M[1], M[2]
    
    # Moments m_k = p^T M_k p
    m0 = p @ M0 @ p
    m2 = p @ M1 @ p
    m4 = p @ M2 @ p
    
    # Moments n_k = 2*pi * q^T M_k q
    factor = 2.0 * np.pi
    n0 = factor * (q @ M0 @ q)
    n2 = factor * (q @ M1 @ q)
    n4 = factor * (q @ M2 @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradients
    # d(m_k)/d(p) = 2 M_k p
    # d(n_k)/d(q) = 2 * factor * M_k q
    
    # Jacobian of p w.r.t a = [a0, a4, a8]
    # p is linear in a.
    J_p = np.zeros((7, 3))
    
    # p0 = a0 + 3a4 + 105a8 + 10395a12
    # a12 = -(a0 + 3a4 + 105a8)/10395
    # p0 = a0 + 3a4 + 105a8 + 10395 * (-(a0 + 3a4 + 105a8)/10395) = 0
    # So p0 is identically 0.
    J_p[0, :] = 0.0
    
    # p1 = -6a4 - 420a8 - 7560a12
    # dp1/da0 = -7560 * (-1/10395) = 7560/10395
    # dp1/da4 = -6 - 7560 * (-3/10395) = -6 + 22680/10395
    # dp1/da8 = -420 - 7560 * (-105/10395) = -420 + 793800/10395
    J_p[1, 0] = 7560.0 / 10395.0
    J_p[1, 1] = -6.0 + 22680.0 / 10395.0
    J_p[1, 2] = -420.0 + 793800.0 / 10395.0
    
    # p2 = a4 + 210a8 + 13860a12
    # dp2/da0 = 13860 * (-1/10395)
    # dp2/da4 = 1 + 13860 * (-3/10395)
    # dp2/da8 = 210 + 13860 * (-105/10395)
    J_p[2, 0] = -13860.0 / 10395.0
    J_p[2, 1] = 1.0 - 41580.0 / 10395.0
    J_p[2, 2] = 210.0 - 1455300.0 / 10395.0
    
    # p3 = -28a8 - 6930a12
    # dp3/da0 = -6930 * (-1/10395)
    # dp3/da4 = -6930 * (-3/10395)
    # dp3/da8 = -28 - 6930 * (-105/10395)
    J_p[3, 0] = 6930.0 / 10395.0
    J_p[3, 1] = 20790.0 / 10395.0
    J_p[3, 2] = -28.0 + 727650.0 / 10395.0
    
    # p4 = a8 + 1155a12
    # dp4/da0 = 1155 * (-1/10395)
    # dp4/da4 = 1155 * (-3/10395)
    # dp4/da8 = 1 + 1155 * (-105/10395)
    J_p[4, 0] = -1155.0 / 10395.0
    J_p[4, 1] = -3465.0 / 10395.0
    J_p[4, 2] = 1.0 - 121275.0 / 10395.0
    
    # p5 = -66a12
    # dp5/da0 = -66 * (-1/10395)
    # dp5/da4 = -66 * (-3/10395)
    # dp5/da8 = -66 * (-105/10395)
    J_p[5, 0] = 66.0 / 10395.0
    J_p[5, 1] = 198.0 / 10395.0
    J_p[5, 2] = 6930.0 / 10395.0
    
    # p6 = a12
    # dp6/da0 = -1/10395
    # dp6/da4 = -3/10395
    # dp6/da8 = -105/10395
    J_p[6, 0] = -1.0 / 10395.0
    J_p[6, 1] = -3.0 / 10395.0
    J_p[6, 2] = -105.0 / 10395.0
    
    # q = D @ p where D = diag([1, -1, 1, -1, 1, -1, 1])
    D = np.diag([1, -1, 1, -1, 1, -1, 1])
    J_q = D @ J_p
    
    # Gradient of m_k w.r.t a:
    # dm_k/da = (dp/da)^T (2 M_k p) = J_p^T (2 M_k p)
    G_m0 = 2.0 * (M0 @ p)
    G_m1 = 2.0 * (M1 @ p)
    
    dm0_da = J_p.T @ G_m0
    dm1_da = J_p.T @ G_m1
    
    # Gradient of n_k w.r.t a:
    # dn_k/da = (dq/da)^T (2 factor M_k q) = J_q^T (2 factor M_k q)
    G_n0 = 2.0 * factor * (M0 @ q)
    G_n1 = 2.0 * factor * (M1 @ q)
    
    dn0_da = J_q.T @ G_n0
    dn1_da = J_q.T @ G_n1
    
    # d(vf)/da = (dm1_da * m0 - m2 * dm0_da) / m0^2
    d_vf_da = (dm1_da * m0 - m2 * dm0_da) / (m0 * m0)
    
    # d(vF)/da = (dn1_da * n0 - n2 * dn0_da) / n0^2
    d_vF_da = (dn1_da * n0 - n2 * dn0_da) / (n0 * n0)
    
    # d(bound)/da = (vF * d_vf_da + vf * d_vF_da) / (2 * bound)
    grad = (vF * d_vf_da + vf * d_vF_da) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    # Use a robust global search followed by local refinement
    # Bounds for a0, a4, a8
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    best_val = np.inf
    best_params = None
    
    # 1. Differential Evolution for global search
    if time.time() < _DEADLINE - 60:
        try:
            res_de = differential_evolution(_objective, bounds, maxiter=100, popsize=15, tol=1e-8, seed=42)
            if res_de.fun < best_val and np.isfinite(res_de.fun):
                best_val = res_de.fun
                best_params = res_de.x.copy()
        except Exception:
            pass
    
    # 2. L-BFGS-B from multiple starts (including DE result)
    starts = []
    if best_params is not None:
        starts.append(best_params)
    
    # Add some heuristic starts
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([1.0, -0.3, 0.0]))
    starts.append(np.array([-1.0, 0.3, 0.0]))
    
    rng = np.random.default_rng(42)
    for i in range(20):
        starts.append(rng.uniform(-2, 2, 3))
    
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
    
    if best_params is None:
        return np.array([0.0, 0.0, 0.0])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
