"""
Combined analytic uncertainty bound: exact gradient + L-BFGS-B, with a global
grid pre-pass to catch basins the seed program may miss, and a tight final polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import lgamma, log, pi

_DEADLINE = time.time() + 195.0

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
# using log-gamma for stability
_I_TABLE = np.zeros(25)
for _n in range(25):
    _log_val = lgamma(2*_n + 1) - _n*log(4.0) - lgamma(_n + 1) + 0.5*log(pi)
    _I_TABLE[_n] = np.exp(_log_val)


def _bound_and_grad(params):
    """
    Compute uncertainty product sqrt(<x^2>_<xi^2>) and its analytic gradient.
    g(x) = P(x) * exp(-x^2/2), P(x) = sum_{k=0}^{6} p_k x^{2k}
    a12 is derived from (a0, a4, a8) to enforce g(0)=0.
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

    # |g|^2 = P(x)^2 * e^{-x^2}
    # m_k = integral x^{2k} P(x)^2 e^{-x^2} dx = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    # We need m_0, m_1 (for <x^2>), and similarly for Fourier side.
    
    # Precompute S_i[k] = sum_{j=0}^{6} p_j I[k+i+j] for i=0..6, k=0..2
    # This avoids redundant inner loops in gradient computation.
    S = np.zeros((7, 3))
    for i in range(7):
        for k in range(3):
            S[i, k] = np.dot(p, _I_TABLE[k+i:k+i+7])

    m = np.array([np.dot(p, S[i, 0]) for i in range(7)])  # m_0
    # Actually m_k = sum_i p_i S[i,k]
    m0 = np.dot(p, S[:, 0])
    m2 = np.dot(p, S[:, 1])  # integral x^2 P^2 e^{-x^2}
    # m4 not needed for the product bound A(f)*A(F) = sqrt(<x^2>_<xi^2>)
    # Wait: A(f)^2 = <x^2> = m2/m0. A(F)^2 = <xi^2> = n2/n0.
    # The bound is sqrt(m2/m0 * n2/n0).

    # Fourier side: G(xi) = sqrt(2*pi) * P(-i*xi) * e^{-xi^2/2}
    # P(-i*xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k}
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    Sq = np.zeros((7, 3))
    for i in range(7):
        for k in range(3):
            Sq[i, k] = np.dot(q, _I_TABLE[k+i:k+i+7])

    n0 = 2.0 * np.pi * np.dot(q, Sq[:, 0])
    n2 = 2.0 * np.pi * np.dot(q, Sq[:, 1])

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Gradient: d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # dm_k/da_c = 2 * sum_i dp_i * S[i, k]  (since S depends on p linearly)
    # Actually dm_k = sum_{i,j} p_i p_j I[k+i+j], so d(dm_k)/da_c = 2*sum_i dp_i * (sum_j p_j I[k+i+j]) = 2*sum_i dp_i S[i,k]

    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

    grad = np.zeros(3)
    for c in range(3):
        dp = np.zeros(7)
        dp[0] = np.array([1.0, 3.0, 105.0])[c] + 945.0*da12[c]
        dp[1] = np.array([0.0, -6.0, -420.0])[c] - 7560.0*da12[c]
        dp[2] = np.array([0.0, 1.0, 210.0])[c] + 13860.0*da12[c]
        dp[3] = np.array([0.0, 0.0, -28.0])[c] - 6930.0*da12[c]
        dp[4] = np.array([0.0, 0.0, 1.0])[c] + 1155.0*da12[c]
        dp[5] = np.array([0.0, 0.0, 0.0])[c] - 66.0*da12[c]
        dp[6] = np.array([0.0, 0.0, 0.0])[c] + 1.0*da12[c]

        dm0 = 2.0 * np.dot(dp, S[:, 0])
        dm2 = 2.0 * np.dot(dp, S[:, 1])

        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        dn0 = 2.0 * 2.0*np.pi * np.dot(dq, Sq[:, 0])
        dn2 = 2.0 * 2.0*np.pi * np.dot(dq, Sq[:, 1])

        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)

        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    return _bound_and_grad(params)


def construct():
    best_val = np.inf
    best_params = None

    def _try_optimize(start, maxiter=3000):
        nonlocal best_val, best_params
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # --- Phase 1: Global grid pre-pass (inspiration from grid-search idea) ---
    # The constraint P(0)=0 reduces the space. We grid over (a4, a8) in a reasonable
    # range and solve for a0 = -945*a12 - 3*a4 - 105*a8... wait, a12 is derived from (a0,a4,a8).
    # Instead, parameterize by (a4, a8) and a free a0, but that's 3D.
    # Better: grid over (a0, a4) with a8=0, and (a0, a8) with a4=0, and (a4, a8) with a0=0?
    # Actually, let's just do a coarse 2D grid over (a4, a8) for a few fixed a0 values,
    # and over (a0, a4) for a few fixed a8 values. This is a cheap global scan.
    
    # Coarse scan: a0 in [-3, 1], a4 in [-0.5, 0.5], a8 in [-0.01, 0.01]
    # We'll sample a0 at 5 points, a4 at 5 points, a8 at 3 points = 75 starts.
    # But that's a lot of L-BFGS runs. Let's be more targeted.
    
    # Seed and nearby points first
    seed = np.array([-1.13, 0.04, 0.000308642])
    _try_optimize(seed, maxiter=2000)
    
    # A handful of structured starts
    structured_starts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([-1.0, 0.1, 0.0]),
        np.array([-1.0, 0.0, 0.001]),
        np.array([-2.0, 0.05, 0.0]),
        np.array([-0.5, 0.02, 0.0]),
        np.array([1.0, -0.05, 0.0]),
        np.array([-1.5, 0.08, 0.001]),
        np.array([-3.0, 0.1, 0.002]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        structured_starts.append(rng.normal(0, 0.5, 3))
    
    for start in structured_starts:
        if time.time() > _DEADLINE - 60:
            break
        _try_optimize(start, maxiter=1500)
    
    # --- Phase 2: Refine best with tighter tolerance ---
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # --- Phase 3: Final Nelder-Mead polish (robust, derivative-free) ---
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # --- Phase 4: One more L-BFGS-B from the Nelder-Mead result ---
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return seed
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
