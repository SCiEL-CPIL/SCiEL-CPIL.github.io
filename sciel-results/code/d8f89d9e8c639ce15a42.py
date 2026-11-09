"""MODEL-AND-OPTIMIZE: compute the refined Cohn-Goncalves Fourier sign-uncertainty
upper bound (C4) from a Laguerre polynomial with 12 positive double roots, and
optimize the root locations to minimize C4.

Structural change vs. seed: use an ANALYTIC evaluation of the bound via the
elementary symmetric polynomials of the roots (recursion on the coefficient
vector of prod (x - r_i)), which is orders of magnitude faster than Gaussian
quadrature. This enables a GLOBAL stochastic search (differential evolution)
plus a local SLSQP polish within the 630 s wall-clock budget.

C4 = int_0^inf x * P(x)^2 e^{-x} dx / int_0^inf P(x)^2 e^{-x} dx,
P(x) = prod_{i=1}^{m} (x - r_i)^2.

With c_k = e_k(r) the elementary symmetric polynomials, and a_k = sum_i r_i^2:
  int P^2 e^{-x} dx = sum_{k=0}^{4m} c_k^2 * k!
  int x P^2 e^{-x} dx = sum_{k=0}^{4m} (c_k^2 * (k+1)! + 2 a_k c_k c_{k-2} * (k-2)!)
(lower index terms with k<0 / k<2 are omitted).
"""
import time
import math
import numpy as np

_START = time.time()
_DEADLINE = 630.0 - 60.0  # safety margin before the 630 s kill


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _c4_from_coeffs(c):
    """Compute C4 from the coefficient vector c of P(x) = sum_k c_k x^k.

    c has length 4m+1 (degree 4m). c[0]=1, c[4m]=prod r_i^2.
    """
    n = len(c) - 1  # degree
    # factorials up to 2n
    fact = np.empty(2 * n + 1, dtype=float)
    fact[0] = 1.0
    for i in range(1, 2 * n + 1):
        fact[i] = fact[i - 1] * i

    c2 = c * c
    den = np.dot(c2, fact[:n + 1])  # sum c_k^2 k!

    # numerator = sum c_k^2 (k+1)! + 2 * a_k * c_k * c_{k-2} * (k-2)!
    num = np.dot(c2, fact[1:n + 2])  # c_k^2 * (k+1)! for k=0..n
    # cross term: for k=2..n, 2 a_k c_k c_{k-2} (k-2)!
    if n >= 2:
        # a_k = sum_{i=1}^{4m-k} c_{i} c_{k-i} for i=1..k-1  (since c_0=1, a_k = sum_{i=1}^{k-1} c_i c_{k-i})
        # but more precisely a_k = sum_{i=1}^{k-1} c_i c_{k-i}  (each unordered pair counted once via i and k-i)
        # Actually a_k = sum_{i=1}^{k-1} c_i c_{k-i}. For k=2: c_1^2. Good.
        ks = np.arange(2, n + 1)
        a_k = np.array([
            np.dot(c[1:k], c[k - 1:0:-1]) for k in ks
        ])
        cross = 2.0 * a_k * c[2:n + 1] * c[0:n - 1] * fact[0:n - 1]
        num = num + np.sum(cross)

    if den <= 0 or not np.isfinite(den):
        return 1e10
    return num / den


def _bound_from_roots(roots):
    """Compute C4 from double-root locations via analytic coefficient recursion."""
    roots = np.asarray(roots, dtype=float)
    m = len(roots)
    # P(x) = prod_{i=1}^{m} (x - r_i)^2. Build by repeated multiplication.
    # Start with [1] (degree 0), multiply by (x - r)^2 = x^2 - 2 r x + r^2.
    c = np.array([1.0], dtype=float)
    for r in roots:
        r2 = r * r
        # new_c[k] = c[k] (from x^2 * c) - 2r * c[k-1] + r^2 * c[k-2]
        nc = np.zeros(len(c) + 2, dtype=float)
        nc[2:] += c           # x^2 * c
        nc[1:-1] -= 2.0 * r * c  # -2r x * c
        nc[0:-2] += r2 * c     # r^2 * c
        c = nc
    return _c4_from_coeffs(c)


def _objective(roots):
    """C4 bound with constraint checking. Lower is better."""
    roots = np.sort(roots)
    if roots.min() <= 1e-6:
        return 1e10
    if np.any(np.diff(roots) < 1e-5):
        return 1e10
    return _bound_from_roots(roots)


def construct():
    """Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1). Score is the upper bound on C4 (lower is better)."""
    seed = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    best = seed.copy()
    best_val = _objective(seed)

    rng = np.random.default_rng(2024)

    # ---- Global search: differential evolution in log-space ----
    def log_obj(logs):
        r = np.exp(logs)
        r = np.sort(r)
        if r.min() <= 1e-6:
            return 1e10
        if np.any(np.diff(r) < 1e-4):
            return 1e10
        return _bound_from_roots(r)

    x0 = np.log(seed).astype(float)

    # Bounds in log-space: roots in [0.1, 200] -> logs in [-2.3, 5.3]
    bounds = [(-2.3, 5.3)] * 12

    try:
        from scipy.optimize import differential_evolution

        def de_callback(xk, convergence):
            nonlocal best, best_val
            v = log_obj(xk)
            if v < best_val:
                r = np.sort(np.exp(xk))
                if r.min() > 1e-4 and np.all(np.diff(r) > 1e-3):
                    best = r.copy()
                    best_val = float(v)
            return _remaining() < 30.0  # stop if little time left

        t0 = time.time()
        de_res = differential_evolution(
            log_obj,
            bounds,
            seed=2024,
            maxiter=400,
            popsize=15,
            tol=1e-8,
            mutation=(0.4, 1.0),
            recombination=0.8,
            polish=False,
            init="latinhypercube",
            callback=de_callback,
            workers=-1,
        )
        if de_res.fun < best_val:
            r = np.sort(np.exp(de_res.x))
            if r.min() > 1e-4 and np.all(np.diff(r) > 1e-3):
                best = r.copy()
                best_val = float(de_res.fun)
    except Exception:
        pass

    # ---- Local polish: SLSQP from the best DE candidate and a few random starts ----
    if _remaining() > 40.0:
        try:
            from scipy.optimize import minimize

            def slsqp_obj(logs):
                r = np.exp(logs)
                r = np.sort(r)
                if r.min() <= 1e-5:
                    return 1e10
                if np.any(np.diff(r) < 1e-4):
                    return 1e10
                return _bound_from_roots(r)

            # Start points: best + a few perturbations
            starts = [np.log(best).astype(float)]
            for _ in range(3):
                cand = np.log(best) + rng.normal(0.0, 0.2, size=12)
                starts.append(cand)

            for s in starts:
                if _remaining() < 25.0:
                    break
                try:
                    res = minimize(
                        slsqp_obj,
                        s,
                        method="SLSQP",
                        bounds=bounds,
                        options={"maxiter": 300, "ftol": 1e-10},
                    )
                    if res.fun < best_val:
                        r = np.sort(np.exp(res.x))
                        if r.min() > 1e-4 and np.all(np.diff(r) > 1e-3):
                            best = r.copy()
                            best_val = float(res.fun)
                except Exception:
                    pass
        except Exception:
            pass

    # ---- Fallback: try scaling the seed down ----
    if best_val >= 1.0:
        for scale in [0.3, 0.5, 0.7, 0.8, 0.9]:
            cand = seed * scale
            val = _objective(cand)
            if val < best_val:
                best = cand.copy()
                best_val = val

    return [float(x) for x in np.sort(best)]
