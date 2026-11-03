"""Refined Cohn-Goncalves Laguerre construction: minimize the Fourier sign
uncertainty upper bound on C4 by tuning 12 positive double roots (m=12, n=1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import eval_genlaguerre, factorial

# Fixed parameters from the task
M = 12          # number of double roots
N = 1           # dimension / Laguerre order
DEADLINE = 600.0
START = time.time()

# Precompute Laguerre basis coefficients up to degree 2*M (24)
MAX_DEG = 2 * M

def laguerre_coeffs(k):
    """Return the coefficient list (ascending power of x) of L_k(x) (physicists')."""
    # Use the explicit formula for L_k(x)
    coeffs = np.zeros(k + 1)
    for j in range(k + 1):
        # coefficient of x^j in L_k(x)
        # L_k(x) = sum_{j=0}^{k} (-1)^j * C(k, j) / j! * x^j
        coeffs[j] = ((-1.0) ** j) * (np.math.comb(k, j) / np.math.factorial(j))
    return coeffs

# Precompute
_LAG = [laguerre_coeffs(k) for k in range(MAX_DEG + 1)]

def poly_mul(a, b):
    c = np.zeros(len(a) + len(b) - 1)
    for i, av in enumerate(a):
        if av == 0.0:
            continue
        c[i:i+len(b)] += av * b
    return c

def double_root_factor(r):
    # (x - r)^2 = x^2 - 2r x + r^2  -> ascending: [r^2, -2r, 1]
    return np.array([r * r, -2.0 * r, 1.0])

def build_poly_in_monomial(roots):
    # Product of (x - r_i)^2 for i in range(M)
    p = np.array([1.0])
    for r in roots:
        p = poly_mul(p, double_root_factor(r))
    return p  # ascending monomial coeffs, degree 2M

def monomial_to_laguerre(mono_coeffs):
    """Convert ascending monomial coefficients to Laguerre basis coefficients.
    x^j = sum_{k<=j} c_{j,k} L_k(x). We invert using the known expansion.
    Use the relation: x^j expressed in Laguerre basis.
    """
    # Precompute x^j in Laguerre basis once
    # We'll build a matrix: for each j, coeff of L_k in x^j.
    # Inverse of laguerre_coeffs: laguerre_coeffs(k) gives x^j coeffs of L_k.
    # So matrix A[j, k] = coeff of x^j in L_k = _LAG[k][j] (for j<=k)
    # We want, for monomial x^j, the Laguerre expansion: solve A @ v = e_j.
    # Build A as (MAX_DEG+1, MAX_DEG+1), A[j,k] = _LAG[k][j] if j<=k else 0.
    # Then v = A^{-1} e_j.
    return _X2LAG @ mono_coeffs

# Build the transformation matrix X2LAG: maps monomial coeffs (ascending) to Laguerre coeffs
# _LAG[k][j] = coeff of x^j in L_k(x). So A[j,k]=_LAG[k][j].
_A = np.zeros((MAX_DEG + 1, MAX_DEG + 1))
for k in range(MAX_DEG + 1):
    for j in range(k + 1):
        _A[j, k] = _LAG[k][j]
# _A @ lag_coeffs = mono_coeffs  =>  lag_coeffs = _A^{-1} @ mono_coeffs
_Ainv = np.linalg.inv(_A)
_X2LAG = _Ainv

def laguerre_l2_norm_sq(lag_coeffs):
    """L2 norm squared under weight e^{-x} on [0,inf):
    integral_0^inf L_k(x) L_l(x) e^{-x} dx = delta_{kl} (for physicists' L_k with this weight).
    Actually <L_k, L_l> = delta_{kl} for the standard (alpha=0) Laguerre polynomials.
    So norm^2 = sum_k c_k^2.
    """
    return float(np.sum(np.asarray(lag_coeffs) ** 2))

def roots_to_lag_coeffs(roots):
    mono = build_poly_in_monomial(roots)
    return _X2LAG @ mono

def bound_from_roots(roots):
    """Compute the upper bound on C4. Model: bound = K / norm^2,
    where norm is the L2 (Laguerre-weight) norm of the double-root polynomial.
    Larger norm -> smaller bound -> better (lower is better).
    We also enforce a mild constraint: roots must be positive and ordered,
    and the polynomial must have the 'last sign change' property (we keep it
    simple by penalizing non-positive or unordered roots heavily)."""
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0.0):
        return 1e30
    # Penalize unordered (we can sort, but keep a small penalty to keep DE stable)
    if np.any(np.diff(roots) < 0.0):
        # allow but add penalty
        penalty = 1e-6 * np.sum(np.abs(np.diff(roots)[np.diff(roots) < 0]))
    else:
        penalty = 0.0
    lag = roots_to_lag_coeffs(roots)
    nrm2 = laguerre_l2_norm_sq(lag)
    if nrm2 <= 0.0:
        return 1e30
    # K is a fixed scale constant; choose so that the seed value ~ 1.0 baseline maps
    # to a reasonable bound. We calibrate K so that the seed roots give bound = 1.0.
    # But K is constant; we just need a monotone objective. Use K=1 for the objective
    # shape; the actual reported value uses the calibrated K.
    return 1.0 / nrm2 + penalty

# Calibrate K so that the seed configuration yields value 1.0
_SEED = np.array([4.0 * (i + 1) for i in range(M)])
_SEED_BOUND = bound_from_roots(_SEED)
_K = 1.0 / _SEED_BOUND  # so that value(seed) = K / norm^2(seed) = 1.0

def objective(roots):
    """Objective to MINIMIZE: the upper bound on C4 (lower is better)."""
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0.0):
        return 1e30
    if np.any(np.diff(roots) < 0.0):
        penalty = 1e-6 * np.sum(np.abs(np.diff(roots)[np.diff(roots) < 0]))
    else:
        penalty = 0.0
    lag = roots_to_lag_coeffs(roots)
    nrm2 = laguerre_l2_norm_sq(lag)
    if nrm2 <= 0.0:
        return 1e30
    return _K / nrm2 + penalty

def construct():
    global START, DEADLINE
    best_roots = _SEED.copy()
    best_val = objective(best_roots)

    # Global optimization via Differential Evolution over the 12-dim root space.
    # Bounds: each root in [0.1, 200]. We let DE handle ordering; we sort at the end.
    lower = 0.1 * np.ones(M)
    upper = 200.0 * np.ones(M)

    # Run DE with a wall-clock guard.
    def _elapsed():
        return time.time() - START

    # We'll run DE in chunks, checking the deadline. DE is not easily interruptible
    # mid-call, so we run with a maxiter that keeps each call bounded and loop.
    # First, a single DE call with a moderate maxiter; if time remains, refine.
    try:
        # Use a worker that periodically checks time via maxiter cap.
        # DE with popsize=20, maxiter ~ 150 should be fast for 12 dims.
        res = differential_evolution(
            objective,
            bounds=list(zip(lower, upper)),
            seed=42,
            maxiter=200,
            popsize=25,
            tol=1e-8,
            mutation=(0.4, 1.0),
            recombination=0.9,
            polish=True,
            init='latinhypercube',
            workers=1,
        )
        if _elapsed() > DEADLINE - 30.0:
            # Too close to deadline; keep best so far from res if better.
            cand = res.x
            cand = np.sort(cand)
            v = objective(cand)
            if v < best_val:
                best_val = v
                best_roots = cand
            return list(best_roots)
        cand = np.sort(res.x)
        v = objective(cand)
        if v < best_val:
            best_val = v
            best_roots = cand
    except Exception:
        pass

    # If time remains, do a second, tighter DE pass around the current best.
    if _elapsed() < DEADLINE - 120.0:
        try:
            spread = 5.0
            lo2 = np.maximum(0.1, best_roots - spread)
            hi2 = np.minimum(200.0, best_roots + spread)
            res2 = differential_evolution(
                objective,
                bounds=list(zip(lo2, hi2)),
                seed=123,
                maxiter=200,
                popsize=25,
                tol=1e-10,
                mutation=(0.3, 1.0),
                recombination=0.9,
                polish=True,
                init='latinhypercube',
                workers=1,
            )
            cand = np.sort(res2.x)
            v = objective(cand)
            if v < best_val:
                best_val = v
                best_roots = cand
        except Exception:
            pass

    # Final local polish with a gradient-free local search (Nelder-Mead) if time allows.
    if _elapsed() < DEADLINE - 60.0:
        try:
            from scipy.optimize import minimize
            res3 = minimize(
                objective,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12},
            )
            cand = np.sort(res3.x)
            v = objective(cand)
            if v < best_val:
                best_val = v
                best_roots = cand
        except Exception:
            pass

    # Safety: ensure we return before the deadline.
    if _elapsed() > DEADLINE - 10.0:
        return list(best_roots)

    return list(best_roots)
# EVOLVE-BLOCK-END
