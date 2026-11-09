"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre

def construct():
    start_time = time.time()
    deadline = 595.0  # safety margin before the 630 s wall-clock kill

    def remaining():
        return deadline - (time.time() - start_time)

    # ------------------------------------------------------------------
    # Objective: 
    # Based on the observation that scaling up roots increased the score (bad),
    # we hypothesize that smaller roots yield a lower score.
    # We minimize the weighted L2 energy as a proxy, but explore a wider range
    # of scales, particularly smaller ones.
    # ------------------------------------------------------------------
    try:
        from numpy.polynomial.laguerre import laggauss
        xq, wq = laggauss(400)
        xq = np.asarray(xq, dtype=float)
        wq = np.asarray(wq, dtype=float)
    except Exception:
        xq = np.linspace(0.0, 200.0, 8000)
        wq = np.exp(-xq) * (200.0 / (8000 - 1))

    def energy(roots):
        r = np.asarray(roots, dtype=float)
        if np.any(r <= 0.0):
            return 1e30
        logQ = 0.0
        for ri in r:
            logQ += 2.0 * np.log(np.abs(xq - ri))
        log_integrand = np.log(xq) + 2.0 * logQ + np.log(wq)
        log_integrand = np.clip(log_integrand, -700.0, 700.0)
        val = np.sum(np.exp(log_integrand))
        if not np.isfinite(val):
            return 1e30
        return float(val)

    # ------------------------------------------------------------------
    # Candidate starting points
    # ------------------------------------------------------------------
    candidates = []
    
    # 1. Natural Laguerre roots
    try:
        r_lag, _ = roots_genlaguerre(12, 1)
        r_lag = np.sort(np.asarray(r_lag, dtype=float))
        candidates.append(r_lag)
    except Exception:
        pass

    # 2. Scaled Laguerre roots (Focus on smaller scales based on hypothesis)
    if candidates:
        base = candidates[0]
        scales = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0]
        for s in scales:
            candidates.append(base * s)
            
    # 3. Equally spaced, scaled
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    for s in [0.1, 0.5, 1.0]:
        candidates.append(seed_roots * s)

    # 4. Very small roots (clustered near 0)
    small_roots = np.linspace(0.1, 1.0, 12)
    candidates.append(small_roots)
    candidates.append(small_roots * 0.5)

    # ------------------------------------------------------------------
    # Evaluate candidates
    # ------------------------------------------------------------------
    best_val = float("inf")
    best_roots = candidates[0].copy()
    
    for c in candidates:
        if remaining() < 5.0:
            break
        v = energy(c)
        if v < best_val:
            best_val = v
            best_roots = c.copy()

    # ------------------------------------------------------------------
    # Local refinement
    # ------------------------------------------------------------------
    def objective(roots_flat):
        r = np.asarray(roots_flat, dtype=float)
        if np.any(r <= 1e-6):
            return 1e30
        return energy(r)

    bounds = [(1e-6, None)] * 12

    # Refine from the best candidate and a few others
    start_points = [best_roots.copy()]
    # Add the smallest scale candidate explicitly to ensure we explore that basin
    if len(candidates) > 1:
        start_points.append(candidates[1].copy())
    if len(candidates) > 2:
        start_points.append(candidates[2].copy())

    for sp in start_points:
        if remaining() < 20.0:
            break
        try:
            res = minimize(
                objective,
                sp,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8,
                         "maxls": 50}
            )
            if res.fun < best_val and np.all(res.x > 1e-6):
                best_val = float(res.fun)
                best_roots = np.sort(res.x).copy()
        except Exception:
            pass
        
    # Final polish
    if remaining() > 10.0:
        try:
            res = minimize(
                objective,
                best_roots,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 10000, "ftol": 1e-14, "gtol": 1e-10,
                         "maxls": 100}
            )
            if res.fun < best_val and np.all(res.x > 1e-6):
                best_val = float(res.fun)
                best_roots = np.sort(res.x).copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final validity
    # ------------------------------------------------------------------
    out = np.sort(np.asarray(best_roots, dtype=float))
    out = np.maximum(out, 1e-6)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1e-6

    return [float(v) for v in out]
# EVOLVE-BLOCK-END
