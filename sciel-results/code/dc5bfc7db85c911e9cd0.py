"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound on A(f)*A(Fourier(f)) = max_{lambda>=0} (1-g(lambda))/g(lambda)
over three free Hermite coefficients (degrees 0, 4, 8). The degree-12 weight is
fixed so that g(0) = 0. Uses critical-point analysis (g'=0 iff P'-lambda*P=0) to
exactly locate the extremal ratio, combined with smooth surrogate optimization.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_START = time.time()
_DEADLINE = 205.0
_MARGIN = 5.0


def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _hermite_phys(n, x):
    """Physicists' Hermite polynomial H_n(x) via recurrence. Vectorized."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _poly_coeffs(a):
    """Return monomial coefficients of P(lambda) = a0*H0 + a4*H4 + a8*H8 + a12*H12.
    
    Physicists' Hermite polynomials:
    H0 = 1
    H4 = 16x^4 - 48x^2 + 12
    H8 = 256x^8 - 4480x^6 + 26880x^4 - 6720x^2 + 1680
    H12 = 4096x^12 - 245760x^10 + 5529600x^8 - 45158400x^6 + 164430720x^4 - 222772500x^2 + 798336
    """
    a0, a4, a8 = a
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 798336.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Coefficients from highest degree to constant
    # H12: 4096, 0, -245760, 0, 5529600, 0, -45158400, 0, 164430720, 0, -222772500, 0, 798336
    # H8:  256, 0, -4480, 0, 26880, 0, -6720, 0, 1680
    # H4:  16, 0, -48, 0, 12
    # H0:  1
    
    coeffs = np.zeros(13)  # degree 12 down to 0
    coeffs[0] = a12 * 4096
    coeffs[2] = a12 * (-245760)
    coeffs[4] = a12 * 5529600
    coeffs[6] = a12 * (-45158400)
    coeffs[8] = a12 * 164430720
    coeffs[10] = a12 * (-222772500)
    coeffs[12] = a12 * 798336
    
    coeffs[4] += a8 * 256
    coeffs[6] += a8 * (-4480)
    coeffs[8] += a8 * 26880
    coeffs[10] += a8 * (-6720)
    coeffs[12] += a8 * 1680
    
    coeffs[8] += a4 * 16
    coeffs[10] += a4 * (-48)
    coeffs[12] += a4 * 12
    
    coeffs[12] += a0 * 1
    
    return coeffs, a12


def _g_of_lambda(a, lam):
    """g(lambda) = P(lambda) * exp(-lambda^2/2), with a12 chosen so g(0) = 0."""
    coeffs, a12 = _poly_coeffs(a)
    lam = np.asarray(lam, dtype=float)
    # Evaluate polynomial using Horner's method
    poly = np.zeros_like(lam)
    for c in coeffs:
        poly = poly * lam + c
    return poly * np.exp(-0.5 * lam * lam)


def _critical_points(a):
    """Find critical points of g where g'(lambda) = 0.
    
    g'(lambda) = [P'(lambda) - lambda*P(lambda)] * exp(-lambda^2/2)
    So critical points satisfy: P'(lambda) - lambda*P(lambda) = 0
    
    This is a degree-12 polynomial in lambda.
    """
    coeffs, a12 = _poly_coeffs(a)
    # P'(lambda) coefficients (derivative)
    # If P = sum c_i * lambda^(12-i), then P' = sum (12-i)*c_i * lambda^(11-i)
    P_prime = np.zeros(12)
    for i in range(12):
        P_prime[i] = (12 - i) * coeffs[i]
    # P_prime - lambda*P:
    # lambda*P = sum c_i * lambda^(13-i)
    # So lambda*P has coefficients: 0, c_0, c_1, ..., c_11, c_12 (degree 13 down to 0)
    # P_prime has degree 11: P_prime[0], ..., P_prime[11]
    
    # Q(lambda) = P'(lambda) - lambda*P(lambda)
    # lambda*P(lambda) = c_0*lambda^13 + c_1*lambda^12 + ... + c_12*lambda
    # P'(lambda) = 12*c_0*lambda^11 + 11*c_1*lambda^10 + ... + c_11
    
    # Q has degree 13 (from -c_0*lambda^13)
    Q = np.zeros(14)  # degree 13 down to 0
    # -lambda*P contributes:
    for i in range(13):
        Q[i] = -coeffs[i]  # -c_i * lambda^(13-i)
    Q[13] = 0  # constant term of -lambda*P is 0
    # P' contributes:
    for i in range(12):
        Q[i + 2] += (12 - i) * coeffs[i]  # P' term (12-i)*c_i * lambda^(11-i) = lambda^(13-(i+2))
    
    # Find real roots of Q
    try:
        roots = np.roots(Q)
    except Exception:
        return np.array([])
    
    # Filter to real roots with lambda >= 0
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            re = float(r.real)
            if re >= -1e-10:
                real_roots.append(max(0.0, re))
    
    return np.array(real_roots)


def _bound_exact(a):
    """Exact max of (1-g)/g over lambda>=0 using critical points + grid backup."""
    # Get critical points
    crits = _critical_points(a)
    
    # Also sample a grid to catch any missed points
    grid = np.linspace(0.0, 30.0, 50001)
    
    # Combine critical points with grid
    if len(crits) > 0:
        points = np.concatenate([grid, crits])
    else:
        points = grid
    
    g = _g_of_lambda(a, points)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.maximum(ratio, 0.0)
    
    val = float(np.max(ratio))
    idx = int(np.argmax(ratio))
    x0 = float(points[idx])
    
    # Local refinement around argmax
    if 0.0 < x0 < 30.0:
        local = np.linspace(max(0.0, x0 - 0.01), x0 + 0.01, 10001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-12, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        rl = np.maximum(rl, 0.0)
        val = max(val, float(np.max(rl)))
    
    return val


def _bound_smooth(a, temperature=0.01):
    """Smooth log-sum-exp surrogate for max (1-g)/g."""
    grid = np.linspace(0.0, 25.0, 30001)
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temperature)
    softmax = m + temperature * math.log(float(np.mean(s)))
    return softmax


def _bound_smooth_tight(a, temperature=0.003):
    """Tighter smooth surrogate for final polish phase."""
    grid = np.linspace(0.0, 25.0, 60001)
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temperature)
    softmax = m + temperature * math.log(float(np.mean(s)))
    return softmax


def construct():
    global _START
    _START = time.time()
    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)
    
    # Phase 1: Multi-start global optimization on smooth surrogate
    rng = np.random.default_rng(42)
    starts = [_SEED.copy()]
    for scale in (0.2, 0.4, 0.8, 1.5, 3.0):
        starts.append(_SEED + scale * 0.01 * rng.standard_normal(3))
    for _ in range(8):
        starts.append(rng.standard_normal(3) * 0.06)
    # Structured starts based on problem structure
    starts.append(np.array([-0.3, 0.01, 0.0002]))
    starts.append(np.array([-0.8, 0.03, 0.0005]))
    starts.append(np.array([-1.5, 0.06, 0.001]))
    starts.append(np.array([0.0, 0.0, 0.0]))
    starts.append(np.array([-0.5, 0.02, 0.0004]))
    starts.append(np.array([-1.13, 0.04, 0.00031]))
    starts.append(np.array([-1.13, 0.035, 0.00028]))
    starts.append(np.array([-1.13, 0.045, 0.00034]))
    
    for s in starts:
        if _remaining() < 8.0:
            break
        try:
            res = minimize(
                _bound_smooth, s, method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-14, "gtol": 1e-10},
            )
        except Exception:
            continue
        cand = np.asarray(res.x, dtype=float)
        if not np.all(np.isfinite(cand)):
            continue
        v = _bound_exact(cand)
        if v < best_val:
            best_val = v
            best_a = cand.copy()
    
    # Phase 2: Polish with tighter smooth surrogate
    if _remaining() > 8.0:
        try:
            res = minimize(
                _bound_smooth_tight, best_a, method="L-BFGS-B",
                options={"maxiter": 400, "ftol": 1e-15, "gtol": 1e-12},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Phase 3: Direct optimization on exact objective using Nelder-Mead
    if _remaining() > 6.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="Nelder-Mead",
                options={"maxiter": 800, "xatol": 1e-10, "fatol": 1e-12, "adaptive": True},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Phase 4: Fine grid polish around best point
    if _remaining() > 4.0:
        step = 0.001
        for i in (-3, -2, -1, 0, 1, 2, 3):
            for j in (-3, -2, -1, 0, 1, 2, 3):
                for k in (-3, -2, -1, 0, 1, 2, 3):
                    if _remaining() < 1.0:
                        break
                    cand = best_a + step * np.array([i, j, k], dtype=float)
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
    
    # Phase 5: Even finer local search
    if _remaining() > 3.0:
        step = 0.0003
        for i in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
            for j in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
                for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
                    if _remaining() < 0.5:
                        break
                    cand = best_a + step * np.array([i, j, k], dtype=float)
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
    
    # Phase 6: Ultra-fine polish
    if _remaining() > 2.0:
        step = 0.0001
        for i in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5):
            for j in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5):
                for k in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5):
                    if _remaining() < 0.3:
                        break
                    cand = best_a + step * np.array([i, j, k], dtype=float)
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
    
    return np.array(best_a, dtype=float)

# EVOLVE-BLOCK-END
