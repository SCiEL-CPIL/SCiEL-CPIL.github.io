"""Rewritten program: explicitly optimize the Hermite coefficients to minimize the
uncertainty bound C4 = A(f)*A(Fourier(f)) via stochastic global search (basin hopping)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, basinhopping


def _g_and_gt(x, coeffs, n):
    """Return g(x) and its Fourier transform g_tilde(x) on grid x.

    g(x) = sum_k c_k H_{4k}(x) e^{-x^2/2};  c_3 chosen so g(0)=0.
    Fourier transform of H_m(x) e^{-x^2/2} is (-i)^m H_m(x) e^{-x^2/2}.
    """
    h = np.exp(-0.5 * x * x)
    c = np.asarray(coeffs, dtype=float).copy()
    if c.size < 4:
        c = np.concatenate([c, np.zeros(4 - c.size)])
    c0, c1, c2 = c[0], c[1], c[2]
    h0 = np.ones_like(x)
    h4 = eval_hermite(4, x)
    h8 = eval_hermite(8, x)
    h12 = eval_hermite(12, x)
    # impose g(0)=0  =>  c0 + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    c3 = -(c0 + c1 * (-12.0) + c2 * 168.0) / 792.0
    g = (c0 * h0 + c1 * h4 + c2 * h8 + c3 * h12) * h
    # Fourier phases: (-i)^m  for m = 0,4,8,12  =>  all +1
    gt = (c0 * h0 + c1 * h4 + c2 * h8 + c3 * h12) * h
    return g, gt


def _bound(coeffs, x):
    g, gt = _g_and_gt(x, coeffs, 4)
    A = np.max(np.abs(g))
    B = np.max(np.abs(gt))
    if A <= 0.0:
        return 0.0
    return A * B


def _make_objective(x):
    def obj(c):
        try:
            return _bound(c, x)
        except Exception:
            return 1e12
    return obj


def construct():
    # adaptive dense grid covering the support of the Hermite-Gaussian
    x = np.linspace(-8.0, 8.0, 24000)
    obj = _make_objective(x)

    # seed from the prior (Goncalves et al. 2017)
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = x0
    best_val = obj(x0)

    def local(c):
        try:
            r = minimize(obj, c, method="Nelder-Mead",
                         options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-12,
                                  "adaptive": True})
            if r.fun < best_val:
                return (r.fun, r.x, 1)
        except Exception:
            pass
        return (best_val, best, 0)

    try:
        bh = basinhopping(obj, x0, niter=300, minenergy_callback=None,
                          minimizer_kwargs={"method": "Nelder-Mead"},
                          stepsize=1.2,
                          accept_test=lambda new, old: new.fun <= old.fun + 1e-9)
        if bh.fun < best_val:
            best = bh.x
            best_val = bh.fun
    except Exception:
        pass

    # final polish from the best point found
    try:
        r = minimize(obj, best, method="Nelder-Mead",
                     options={"maxiter": 8000, "xatol": 1e-12, "fatol": 1e-14,
                              "adaptive": True})
        if r.fun < best_val:
            best = r.x
            best_val = r.fun
    except Exception:
        pass

    return np.array(best)
# EVOLVE-BLOCK-END
