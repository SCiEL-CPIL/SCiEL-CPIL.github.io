# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0

# Precompute integrals and matrices
_sp = np.sqrt(np.pi)
_I = np.zeros(13)
_I[0] = _sp
for _n in range(1, 13):
    _I[_n] = _I[_n - 1] * (2 * _n - 1) / 2.0

_M = np.zeros((3, 7, 7))
for _k in range(3):
    for _i in range(7):
        for _j in range(7):
            _idx = _k + _i + _j
            if _idx < 13:
                _M[_k, _i, _j] = _I[_idx]

_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Coefficients for a12 in terms of a0, a4, a8
# a12 = -(a0 + 3*a4 + 105*a8) / 945
_A12_COEF = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])


def _compute_bound_and_grad(a0, a4, a8):
    """Compute the uncertainty bound and its gradient with respect to (a0, a4, a8)."""
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # dp/da for a0, a4, a8
    # p0 = a0 + 3a4 + 105a8 + 945a12, a12 = -(a0+3a4+105a8)/945
    # p0 = a0 + 3a4 + 105a8 - (a0+3a4+105a8) = 0
    # p1 = -6a4 - 420a8 - 7560a12 = -6a4 - 420a8 + 7560(a0+3a4+105a8)/945
    #    = -6a4 - 420a8 + 8(a0+3a4+105a8) = 8a0 + (-6+24)a4 + (-420+840)a8 = 8a0 + 18a4 + 420a8
    # p2 = a4 + 210a8 + 13860a12 = a4 + 210a8 - 13860(a0+3a4+105a8)/945
    #    = a4 + 210a8 - 14.6667(a0+3a4+105a8) = -14.6667a0 + (1-44)a4 + (210-1540)a8
    #    = -14.6667a0 - 43a4 - 1330a8
    # p3 = -28a8 - 6930a12 = -28a8 + 6930(a0+3a4+105a8)/945 = 7.3333a0 + 22a4 + (770-28)a8 = 7.3333a0+22a4+742a8
    # p4 = a8 + 1155a12 = a8 - 1155(a0+3a4+105a8)/945 = -1.2222a0 - 3.6667a4 + (1-127.5)a8 = -1.2222a0-3.6667a4-126.5a8
    # p5 = -66a12 = 66(a0+3a4+105a8)/945 = 0.06984a0 + 0.20952a4 + 7.3333a8
    # p6 = a12 = -(a0+3a4+105a8)/945 = -0.001058a0 - 0.003175a4 - 0.11111a8

    dp = np.zeros((7, 3))
    # p0 = 0
    dp[0] = [0.0, 0.0, 0.0]
    # p1 = 8a0 + 18a4 + 420a8
    dp[1] = [8.0, 18.0, 420.0]
    # p2 = -14.6667a0 - 43a4 - 1330a8
    dp[2] = [-14.666667, -43.0, -1330.0]
    # p3 = 7.3333a0 + 22a4 + 742a8
    dp[3] = [7.333333, 22.0, 742.0]
    # p4 = -1.2222a0 - 3.6667a4 - 126.5a8
    dp[4] = [-1.222222, -3.666667, -126.5]
    # p5 = 0.06984a0 + 0.20952a4 + 7.3333a8
    dp[5] = [66.0/945.0, 198.0/945.0, 6930.0/945.0]
    # p6 = -1/945 a0 - 3/945 a4 - 105/945 a8
    dp[6] = [-1.0/945.0, -3.0/945.0, -105.0/945.0]

    m = np.einsum('i,kij,j->k', p, _M, p)
    q = p * _SIGNS
    n = np.einsum('i,kij,j->k', q, _M, q) * 2.0 * np.pi

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18, np.zeros(3)

    bound = np.sqrt(vf * vF)

    # Gradient computation
    # bound = sqrt(vf * vF)
    # d(bound)/da = 1/(2*bound) * (vF * d(vf)/da + vf * d(vF)/da)
    # vf = m2/m0, d(vf)/da = (m0*dm2/da - m2*dm0/da)/m0^2
    # vF = n2/n0, d(vF)/da = (n0*dn2/da - n2*dn0/da)/n0^2

    # dm_k/da_j = 2 * sum_i p_i * M[k,i,j] * dp_j/da... wait
    # m_k = sum_{i,j} p_i M[k,i,j] p_j
    # dm_k/da_a = sum_{i,j} (dp_i/da_a * M[k,i,j] * p_j + p_i * M[k,i,j] * dp_j/da_a)
    #           = 2 * sum_{i,j} dp_i/da_a * M[k,i,j] * p_j  (by symmetry of M in i,j)

    grad_bound = np.zeros(3)
    for a in range(3):
        dm = np.zeros(3)
        dn = np.zeros(3)
        for k in range(3):
            # dm_k/da_a = 2 * dp[:,a]^T M[k] p
            dm[k] = 2.0 * np.dot(dp[:, a], np.dot(_M[k], p))
            dn[k] = 2.0 * np.dot(dp[:, a] * _SIGNS, np.dot(_M[k], q)) * 2.0 * np.pi

        dm0, dm2 = dm[0], dm[1]
        dn0, dn2 = dn[0], dn[1]

        dvf = (m0 * dm2 - m2 * dm0) / (m0 * m0)
        dvF = (n0 * dn2 - n2 * dn0) / (n0 * n0)

        grad_bound[a] = 0.5 / bound * (vF * dvf + vf * dvF)

    return bound, grad_bound


def _objective(x):
    b, _ = _compute_bound_and_grad(x[0], x[1], x[2])
    return b


def _objective_grad(x):
    b, g = _compute_bound_and_grad(x[0], x[1], x[2])
    return b, g


def construct():
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _objective(seed)

    rng = np.random.default_rng(42)

    # Multi-start L-BFGS-B optimization
    best_bound = seed_bound
    best_params = seed.copy()

    # Generate multiple starting points
    starts = [seed.copy()]

    # Random starts in various ranges
    for r in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        for _ in range(20):
            t1 = rng.uniform(-np.pi, np.pi)
            t2 = rng.uniform(0, np.pi)
            st2 = np.sin(t2)
            x = np.array([
                r * st2 * np.cos(t1),
                r * st2 * np.sin(t1),
                r * np.cos(t2)
            ])
            starts.append(x)

    # Also try some structured starts
    for a0 in [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]:
        for a4 in [-2, -1, -0.5, 0, 0.5, 1, 2]:
            for a8 in [-1, -0.5, 0, 0.5, 1]:
                starts.append(np.array([a0, a4, a8]))

    for x0 in starts:
        if time.time() > _DEADLINE - 10:
            break

        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success and np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final verification
    final_bound = _objective(best_params)
    if not np.isfinite(final_bound) or final_bound > seed_bound:
        return seed.copy()
    return best_params
# EVOLVE-BLOCK-END
