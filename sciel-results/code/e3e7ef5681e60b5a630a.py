"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

# Precompute quadrature nodes (standard precision for search, high precision for polish)
_NQ_STD = 200
_NQ_HIGH = 400
_xq_std, _wq_std = np.polynomial.laguerre.laggauss(_NQ_STD)
_xq_high, _wq_high = np.polynomial.laguerre.laggauss(_NQ_HIGH)


def _c4_from_roots(roots, xq, wq):
    """Compute C4 = int(P^2 w)/(int(P w))^2 for P = prod (x - r_i)^2."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf

    diff = xq[:, np.newaxis] - roots[np.newaxis, :]
    log_abs = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs, axis=1)

    P = np.exp(log_P)
    P2 = P * P

    int_P = np.sum(wq * P)
    int_P2 = np.sum(wq * P2)

    if not np.isfinite(int_P) or not np.isfinite(int_P2) or int_P <= 0:
        return np.inf

    return int_P2 / (int_P * int_P)


def _objective_roots(roots):
    """Objective in raw (positive) root space — used by differential evolution."""
    return _c4_from_roots(roots, _xq_std, _wq_std)


def _objective_log(t):
    """Objective in log-space (x = exp(t)) — unconstrained, for L-BFGS-B polish."""
    x = np.exp(t)
    return _c4_from_roots(x, _xq_std, _wq_std)


def _polish(roots, xq, wq, maxiter=2000, ftol=1e-16, gtol=1e-14):
    """Local L-BFGS-B polish in log-space; returns (improved_roots, improved_score)."""
    best_roots = np.asarray(roots, dtype=np.float64)
    best_score = _c4_from_roots(best_roots, xq, wq)
    try:
        t0 = np.log(np.maximum(best_roots, 1e-300))
        res = minimize(_objective_log, t0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        cand = np.exp(res.x)
        cand = np.sort(cand)
        sc = _c4_from_roots(cand, xq, wq)
        if np.all(cand > 0) and np.isfinite(sc) and sc < best_score:
            best_roots, best_score = cand, sc
    except Exception:
        pass
    return best_roots, best_score


def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580.0  # safety margin before 630 s kill

    # Strong initial guess: zeros of Laguerre L_12 (equilibrium of the log-gas).
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)

    best_roots = x0.copy()
    best_score = _c4_from_roots(best_roots, _xq_std, _wq_std)

    if time.time() > deadline - 150:
        return [float(v) for v in best_roots]

    rng = np.random.default_rng(42)

    # ---- Build diverse initial points (raw positive space) ----
    initial_points = [x0.copy()]

    # Perturbed Laguerre roots (multiplicative, log-space noise)
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        p = x0 * np.exp(scale * rng.standard_normal(12))
        initial_points.append(np.sort(np.maximum(p, 1e-4)))

    # Scaled Laguerre roots
    for s in [0.6, 0.8, 1.2, 1.5, 2.0, 3.0]:
        initial_points.append(np.sort(x0 * s))

    # Log-uniform grids (geometric in linear space) — productive family
    for ls in [-2.0, -1.5, -1.0, -0.5, 0.0]:
        for le in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if le > ls:
                rr = np.exp(np.linspace(ls, le, 12))
                if rr[0] > 1e-3 and rr[-1] < 200:
                    initial_points.append(rr.copy())

    # Random log-uniform points
    for _ in range(6):
        rr = np.exp(rng.uniform(-2.0, 3.0, size=12))
        initial_points.append(np.sort(rr))

    # ---- Local polish (log-space) from each initial point ----
    for x_init in initial_points:
        if time.time() > deadline - 90:
            break
        roots, sc = _polish(x_init, _xq_std, _wq_std, maxiter=800, ftol=1e-15, gtol=1e-13)
        if sc < best_score:
            best_score, best_roots = sc, roots

    # ---- Global search: differential evolution in raw positive space ----
    if time.time() < deadline - 60:
        try:
            de_bounds = [(1e-3, 200.0)] * 12
            res_de = differential_evolution(
                _objective_roots,
                bounds=de_bounds,
                maxiter=60,
                popsize=18,
                tol=1e-10,
                seed=42,
                polish=False,
                updating='immediate',
                workers=1,
            )
            cand = np.sort(np.asarray(res_de.x, dtype=np.float64))
            if np.all(cand > 0) and not np.any(np.isnan(cand)):
                roots, sc = _polish(cand, _xq_std, _wq_std, maxiter=1000, ftol=1e-15, gtol=1e-13)
                if sc < best_score:
                    best_score, best_roots = sc, roots
        except Exception:
            pass

    # ---- Final high-precision polish (more quadrature nodes, tighter tol) ----
    if time.time() < deadline - 10:
        roots, sc = _polish(best_roots, _xq_high, _wq_high,
                            maxiter=3000, ftol=1e-17, gtol=1e-15)
        if sc < best_score:
            best_score, best_roots = sc, roots

    # ---- Guarantee valid output ----
    result = [max(float(v), 1e-8) for v in best_roots]
    if len(result) != 12 or any(not np.isfinite(v) for v in result):
        result = [float(v) for v in x0]
    return result
# EVOLVE-BLOCK-END
