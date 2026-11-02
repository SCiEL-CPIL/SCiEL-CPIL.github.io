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
    # Use a finer grid for better accuracy in the max computation
    # The Hermite-Gaussian decays as e^{-x^2/2}, so [-6, 6] is sufficient
    # Use more points for better resolution of the maxima
    x = np.linspace(-6.0, 6.0, 40000)
    obj = _make_objective(x)

    # Multiple starting points to explore different regions
    starting_points = [
        np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0]),  # prior
        np.array([-1.0, 0.0, 0.0]),
        np.array([-0.5, 0.04, 0.0]),
        np.array([-1.5, 0.0, 0.001]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-0.8, 0.03, 0.0001]),
    ]

    best = starting_points[0]
    best_val = obj(best)

    # First, do a local optimization from each starting point
    for x0 in starting_points:
        try:
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-12,
                                  "adaptive": True})
            if r.fun < best_val:
                best = r.x.copy()
                best_val = r.fun
        except Exception:
            pass

    # Now use basin hopping from the best point found
    def local(c):
        try:
            r = minimize(obj, c, method="Nelder-Mead",
                         options={"maxiter": 3000, "xatol": 1e-11, "fatol": 1e-13,
                                  "adaptive": True})
            if r.fun < best_val:
                return (r.fun, r.x, 1)
        except Exception:
            pass
        return (best_val, best, 0)

    try:
        bh = basinhopping(obj, best, niter=200, minenergy_callback=None,
                          minimizer_kwargs={"method": "Nelder-Mead"},
                          stepsize=0.8,
                          accept_test=lambda new, old: new.fun <= old.fun + 1e-9)
        if bh.fun < best_val:
            best = bh.x.copy()
            best_val = bh.fun
    except Exception:
        pass

    # Final polish with very tight tolerances
    try:
        r = minimize(obj, best, method="Nelder-Mead",
                     options={"maxiter": 10000, "xatol": 1e-13, "fatol": 1e-15,
                              "adaptive": True})
        if r.fun < best_val:
            best = r.x.copy()
            best_val = r.fun
    except Exception:
        pass

    # One more basin hopping from the polished point with smaller steps
    try:
        bh2 = basinhopping(obj, best, niter=100, minenergy_callback=None,
                           minimizer_kwargs={"method": "Nelder-Mead",
                                             "options": {"maxiter": 2000,
                                                         "xatol": 1e-12,
                                                         "fatol": 1e-14,
                                                         "adaptive": True}},
                           stepsize=0.3,
                           accept_test=lambda new, old: new.fun <= old.fun + 1e-10)
        if bh2.fun < best_val:
            best = bh2.x.copy()
            best_val = bh2.fun
    except Exception:
        pass

    # Final ultra-fine polish
    try:
        r = minimize(obj, best, method="Nelder-Mead",
                     options={"maxiter": 15000, "xatol": 1e-14, "fatol": 1e-16,
                              "adaptive": True})
        if r.fun < best_val:
            best = r.x.copy()
            best_val = r.fun
    except Exception:
        pass

    return np.array(best)
# EVOLVE-BLOCK-END
