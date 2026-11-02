"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound on A(f)*A(Fourier(f)) = max_{lambda>=0} (1-g(lambda))/g(lambda)
over three free Hermite coefficients (degrees 0, 4, 8). The degree-12 weight is
fixed so that g(0) = 0. Uses adaptive grid with critical point analysis.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, brentq

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_START = time.time()
_DEADLINE = 210.0
_MARGIN = 30.0


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


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2),
    with a12 chosen so g(0) = 0."""
    a0, a4, a8 = a
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 798336.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    lam = np.asarray(lam, dtype=float)
    h0 = np.ones_like(lam)
    h4 = _hermite_phys(4, lam)
    h8 = _hermite_phys(8, lam)
    h12 = _hermite_phys(12, lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _g_prime_of_lambda(a, lam):
    """Derivative of g(lambda) with respect to lambda."""
    a0, a4, a8 = a
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 798336.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    lam = np.asarray(lam, dtype=float)
    h0 = np.ones_like(lam)
    h4 = _hermite_phys(4, lam)
    h8 = _hermite_phys(8, lam)
    h12 = _hermite_phys(12, lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    
    # g'(lambda) = P'(lambda)*exp(-lambda^2/2) - lambda*P(lambda)*exp(-lambda^2/2)
    # P'(lambda) = a0*H0' + a4*H4' + a8*H8' + a12*H12'
    # H_n'(x) = 2n*H_{n-1}(x)
    h0_prime = np.zeros_like(lam)
    h4_prime = 2 * 4 * _hermite_phys(3, lam)
    h8_prime = 2 * 8 * _hermite_phys(7, lam)
    h12_prime = 2 * 12 * _hermite_phys(11, lam)
    poly_prime = a0 * h0_prime + a4 * h4_prime + a8 * h8_prime + a12 * h12_prime
    
    return (poly_prime - lam * poly) * np.exp(-0.5 * lam * lam)


def _find_critical_points(a, lam_min=0.01, lam_max=30.0, n_grid=2000):
    """Find critical points of g(lambda) where g'(lambda)=0."""
    grid = np.linspace(lam_min, lam_max, n_grid)
    g_prime = _g_prime_of_lambda(a, grid)
    
    # Find sign changes
    signs = np.sign(g_prime)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    
    critical_points = []
    for idx in sign_changes:
        a_left = grid[idx]
        a_right = grid[idx + 1]
        try:
            root = brentq(lambda x: _g_prime_of_lambda(a, np.array([x]))[0], a_left, a_right, xtol=1e-12)
            if lam_min <= root <= lam_max:
                critical_points.append(root)
        except Exception:
            pass
    
    return critical_points


def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 on a dense grid with critical point analysis."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.001, 25.0, 100001),
            np.geomspace(1e-4, 50.0, 5000),
        ])
    
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-15, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.maximum(ratio, 0.0)
    
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    
    # Evaluate at critical points
    try:
        crit_points = _find_critical_points(a)
        for cp in crit_points:
            g_cp = _g_of_lambda(a, np.array([cp]))[0]
            if g_cp > 1e-15:
                r_cp = (1.0 - g_cp) / g_cp
                if np.isfinite(r_cp) and r_cp > val:
                    val = float(r_cp)
    except Exception:
        pass
    
    # Local refinement around argmax
    if 0.001 < x0 < 50.0:
        local = np.linspace(max(0.001, x0 - 0.02), x0 + 0.02, 5001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-15, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        rl = np.maximum(rl, 0.0)
        val = max(val, float(np.max(rl)))
    
    return val


def _bound_smooth(a, grid=None, temperature=0.01):
    """Smooth log-sum-exp surrogate for max (1-g)/g."""
    if grid is None:
        grid = np.linspace(0.001, 25.0, 50001)
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-15, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temperature)
    softmax = m + temperature * math.log(float(np.mean(s)))
    return softmax


def _bound_smooth_adaptive(a, grid=None, temperature=0.005):
    """Adaptive smooth surrogate with finer grid near the maximum."""
    if grid is None:
        grid = np.linspace(0.001, 25.0, 30000)
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-15, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    
    # Find approximate argmax and refine
    idx = int(np.argmax(ratio))
    x0 = float(grid[idx])
    
    # Add fine grid around argmax
    local_grid = np.linspace(max(0.0005, x0 - 0.1), x0 + 0.1, 10000)
    gl = _g_of_lambda(a, local_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        rl = (1.0 - gl) / gl
    rl = np.where(gl > 1e-15, rl, 1e18)
    rl = np.where(np.isfinite(rl), rl, 1e18)
    rl = np.clip(rl, 1e-12, 1e15)
    
    combined = np.concatenate([ratio, rl])
    m = float(np.max(combined))
    s = np.exp((combined - m) / temperature)
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
    for scale in (0.1, 0.2, 0.4, 0.8, 1.5, 3.0):
        starts.append(_SEED + scale * 0.005 * rng.standard_normal(3))
    for _ in range(20):
        starts.append(rng.standard_normal(3) * 0.05)
    # Structured starts based on problem structure
    starts.append(np.array([-1.0, 0.035, 0.00025]))
    starts.append(np.array([-1.2, 0.045, 0.00035]))
    starts.append(np.array([-0.8, 0.025, 0.0002]))
    starts.append(np.array([-1.5, 0.06, 0.0005]))
    starts.append(np.array([-0.5, 0.02, 0.00015]))
    starts.append(np.array([-1.1, 0.04, 0.0003]))
    starts.append(np.array([-0.9, 0.03, 0.00022]))
    starts.append(np.array([-1.3, 0.05, 0.0004]))
    starts.append(np.array([-1.4, 0.055, 0.00045]))
    starts.append(np.array([-0.7, 0.022, 0.00018]))
    
    for s in starts:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(
                _bound_smooth_adaptive, s, method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-11},
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
    if _remaining() > 5.0:
        try:
            res = minimize(
                _bound_smooth_adaptive, best_a, method="L-BFGS-B",
                options={"maxiter": 800, "ftol": 1e-15, "gtol": 1e-13},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead on exact objective for fine-tuning
    if _remaining() > 8.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-12, "fatol": 1e-14, "adaptive": True},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Phase 4: Coordinate descent with line search
    if _remaining() > 6.0:
        # Try optimizing each coordinate separately
        for dim in range(3):
            if _remaining() < 3.0:
                break
            # Find optimal value for this coordinate
            lo = best_a[dim] - 0.01
            hi = best_a[dim] + 0.01
            # Sample to find bracket
            samples = np.linspace(lo, hi, 100)
            vals = []
            for s in samples:
                cand = best_a.copy()
                cand[dim] = s
                v = _bound_exact(cand)
                vals.append(v)
            idx_min = int(np.argmin(vals))
            # Refine around minimum
            if 0 < idx_min < 99:
                lo2 = samples[idx_min - 1]
                hi2 = samples[idx_min + 1]
                # Golden section search
                gr = (math.sqrt(5) - 1) / 2
                for _ in range(50):
                    if _remaining() < 1.0:
                        break
                    x1 = hi2 - gr * (hi2 - lo2)
                    x2 = lo2 + gr * (hi2 - lo2)
                    cand1 = best_a.copy()
                    cand1[dim] = x1
                    v1 = _bound_exact(cand1)
                    cand2 = best_a.copy()
                    cand2[dim] = x2
                    v2 = _bound_exact(cand2)
                    if v1 < v2:
                        hi2 = x2
                    else:
                        lo2 = x1
                # Final evaluation
                x_opt = (lo2 + hi2) / 2
                cand = best_a.copy()
                cand[dim] = x_opt
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
    
    # Phase 5: Fine grid polish around best point
    if _remaining() > 3.0:
        step = 0.0005
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
    
    # Phase 6: Even finer local search
    if _remaining() > 2.0:
        step = 0.00015
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
    
    # Phase 7: Very fine polish
    if _remaining() > 1.0:
        step = 0.00005
        for i in (-2, -1, 0, 1, 2):
            for j in (-2, -1, 0, 1, 2):
                for k in (-2, -1, 0, 1, 2):
                    if _remaining() < 0.3:
                        break
                    cand = best_a + step * np.array([i, j, k], dtype=float)
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
    
    return np.array(best_a, dtype=float)

# EVOLVE-BLOCK-END
