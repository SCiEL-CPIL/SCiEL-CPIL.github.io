"""Analytic Hermite-Gaussian uncertainty bound with gradient-based L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200  # 10s safety margin before the 210s kill


def _bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty product and its gradient with respect to (a0, a4, a8).
    a12 is determined by the constraint g(0) = 0: a12 = -(a0 + 3*a4 + 105*a8)/945.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Moments of |g|^2
    # m0 = sqrt(pi) * (a0^2 + 3*a4^2 + 105*a8^2 + 945*a12^2)
    # m2 = sqrt(pi) * [a0*(3*a0+12*a4) + a4*(12*a0+105*a4+840*a8) + a8*(840*a4+2940*a8) + 11340*a8*a12]
    # 
    # Moments of |G|^2
    # g0 = a0 + 3*a4 + 105*a8 + 945*a12 = 0 (by constraint)
    # g2 = a0 - 3*a4 + 105*a8 - 945*a12
    # g4 = a0 + 3*a4 + 105*a8 + 945*a12 = 0 (by constraint)
    # Wait, let me recompute g0 and g4.
    # From the code: g0 = a0 + 3*a4 + 105*a8 + 945*a12, and a12 = -(a0+3*a4+105*a8)/945
    # So g0 = a0 + 3*a4 + 105*a8 + 945*(-(a0+3*a4+105*a8)/945) = 0
    # g4 = a0 + 3*a4 + 105*a8 + 945*a12 = 0 as well (same expression)
    # g2 = a0 - 3*a4 + 105*a8 - 945*a12
    #    = a0 - 3*a4 + 105*a8 - 945*(-(a0+3*a4+105*a8)/945)
    #    = a0 - 3*a4 + 105*a8 + a0 + 3*a4 + 105*a8
    #    = 2*a0 + 210*a8

    # So n0 = 2*pi * (g0^2 + 9*g2^2 + 11025*g4^2) = 2*pi * 9 * g2^2 = 18*pi * g2^2
    # n2 = 2*pi * (3*g0^2 + 21*g2^2 + 2940*g4^2) = 2*pi * 21 * g2^2 = 42*pi * g2^2

    g2 = 2.0 * a0 + 210.0 * a8

    sqrt_pi = np.sqrt(np.pi)

    # m0
    m0_inner = a0 * a0 + 3.0 * a4 * a4 + 105.0 * a8 * a8 + 945.0 * a12 * a12
    m0 = sqrt_pi * m0_inner

    # m2
    m2_inner = (a0 * (3.0 * a0 + 12.0 * a4)
                + a4 * (12.0 * a0 + 105.0 * a4 + 840.0 * a8)
                + a8 * (840.0 * a4 + 2940.0 * a8)
                + 11340.0 * a8 * a12)
    m2 = sqrt_pi * m2_inner

    # n0, n2
    n0 = 18.0 * np.pi * g2 * g2
    n2 = 42.0 * np.pi * g2 * g2

    if m0 <= 0.0 or n0 <= 0.0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([0.0, 0.0, 0.0])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0.0 or vF <= 0.0:
        return 1e18, np.array([0.0, 0.0, 0.0])

    bound = np.sqrt(vf * vF)

    # Gradient computation
    # bound = sqrt(vf * vF)
    # dbound/dx = 1/(2*bound) * (vF * dvf/dx + vf * dvF/dx)
    # vf = m2/m0, dvf/dx = (dm2/dx * m0 - m2 * dm0/dx) / m0^2
    # vF = n2/n0, dvF/dx = (dn2/dx * n0 - n2 * dn0/dx) / n0^2

    # dm0/dx: m0 = sqrt_pi * (a0^2 + 3*a4^2 + 105*a8^2 + 945*a12^2)
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    dm0_da0 = sqrt_pi * (2.0 * a0 + 2.0 * 945.0 * a12 * (-1.0 / 945.0))
    dm0_da4 = sqrt_pi * (6.0 * a4 + 2.0 * 945.0 * a12 * (-3.0 / 945.0))
    dm0_da8 = sqrt_pi * (210.0 * a8 + 2.0 * 945.0 * a12 * (-105.0 / 945.0))

    # dm2/dx: m2 = sqrt_pi * m2_inner
    # m2_inner = a0*(3*a0+12*a4) + a4*(12*a0+105*a4+840*a8) + a8*(840*a4+2940*a8) + 11340*a8*a12
    # Let's expand:
    # = 3*a0^2 + 12*a0*a4 + 12*a0*a4 + 105*a4^2 + 840*a4*a8 + 840*a4*a8 + 2940*a8^2 + 11340*a8*a12
    # = 3*a0^2 + 24*a0*a4 + 105*a4^2 + 1680*a4*a8 + 2940*a8^2 + 11340*a8*a12
    #
    # dm2_inner/da0 = 6*a0 + 24*a4
    # dm2_inner/da4 = 24*a0 + 210*a4 + 1680*a8
    # dm2_inner/da8 = 1680*a4 + 5880*a8 + 11340*a12 + 11340*a8*da12/da8
    #               = 1680*a4 + 5880*a8 + 11340*a12 + 11340*a8*(-105/945)
    #               = 1680*a4 + 5880*a8 + 11340*a12 - 1260*a8
    #               = 1680*a4 + 4620*a8 + 11340*a12

    dm2_inner_da0 = 6.0 * a0 + 24.0 * a4
    dm2_inner_da4 = 24.0 * a0 + 210.0 * a4 + 1680.0 * a8
    dm2_inner_da8 = 1680.0 * a4 + 4620.0 * a8 + 11340.0 * a12

    dm2_da0 = sqrt_pi * dm2_inner_da0
    dm2_da4 = sqrt_pi * dm2_inner_da4
    dm2_da8 = sqrt_pi * dm2_inner_da8

    # dn0/dx: n0 = 18*pi*g2^2, g2 = 2*a0 + 210*a8
    # dn0/da0 = 18*pi*2*g2*2 = 72*pi*g2
    # dn0/da4 = 0
    # dn0/da8 = 18*pi*2*g2*210 = 7560*pi*g2
    dn0_da0 = 72.0 * np.pi * g2
    dn0_da4 = 0.0
    dn0_da8 = 7560.0 * np.pi * g2

    # dn2/dx: n2 = 42*pi*g2^2
    # dn2/da0 = 42*pi*2*g2*2 = 168*pi*g2
    # dn2/da4 = 0
    # dn2/da8 = 42*pi*2*g2*210 = 17640*pi*g2
    dn2_da0 = 168.0 * np.pi * g2
    dn2_da4 = 0.0
    dn2_da8 = 17640.0 * np.pi * g2

    # dvf/dx = (dm2/dx * m0 - m2 * dm0/dx) / m0^2
    dvf_da0 = (dm2_da0 * m0 - m2 * dm0_da0) / (m0 * m0)
    dvf_da4 = (dm2_da4 * m0 - m2 * dm0_da4) / (m0 * m0)
    dvf_da8 = (dm2_da8 * m0 - m2 * dm0_da8) / (m0 * m0)

    # dvF/dx = (dn2/dx * n0 - n2 * dn0/dx) / n0^2
    dvF_da0 = (dn2_da0 * n0 - n2 * dn0_da0) / (n0 * n0)
    dvF_da4 = (dn2_da4 * n0 - n2 * dn0_da4) / (n0 * n0)
    dvF_da8 = (dn2_da8 * n0 - n2 * dn0_da8) / (n0 * n0)

    # dbound/dx = 1/(2*bound) * (vF * dvf/dx + vf * dvF/dx)
    grad_da0 = 0.5 / bound * (vF * dvf_da0 + vf * dvF_da0)
    grad_da4 = 0.5 / bound * (vF * dvf_da4 + vf * dvF_da4)
    grad_da8 = 0.5 / bound * (vF * dvf_da8 + vf * dvF_da8)

    return bound, np.array([grad_da0, grad_da4, grad_da8])


def _objective(params):
    val, _ = _bound_and_grad(params[0], params[1], params[2])
    return val if np.isfinite(val) else 1e18


def _objective_grad(params):
    val, grad = _bound_and_grad(params[0], params[1], params[2])
    if not np.isfinite(val):
        return 1e18, np.array([0.0, 0.0, 0.0])
    return val, grad


def construct():
    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240.0])
    best_val, _ = _bound_and_grad(*seed)
    best = seed.copy()

    if not np.isfinite(best_val):
        best_val = 1e18

    rng = np.random.default_rng(42)

    # Multi-start with L-BFGS-B
    starts = [seed.copy()]
    for i in range(15):
        scale = 0.2 / (i + 1)
        starts.append(best + rng.normal(0.0, scale, 3))

    for s in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                           options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # High-precision polish
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best, method='L-BFGS-B',
                           options={'maxiter': 20000, 'gtol': 1e-16, 'ftol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    return np.array(best)
# EVOLVE-BLOCK-END
