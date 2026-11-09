"""MODEL-AND-OPTIMIZE: compute the actual refined Cohn-Goncalves Fourier
sign-uncertainty upper bound (C4) from the Laguerre polynomial with 12 double
roots, and optimize the root locations to minimize it. Lower raw value is
better; the seed ([4,8,...,48]) scores 1.0."""
import time
import numpy as np

_START = time.time()
_DEADLINE = 630.0 - 60.0  # safety margin before the 630 s kill


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _bound_from_roots(roots, m=12):
    """Compute the upper bound on C4 from the given double-root locations.
    
    The refined Cohn-Goncalves bound: C4 = int x*P^2*e^{-x} dx / int P^2*e^{-x} dx
    where P(x) = prod_{i=1}^{m} (x - r_i)^2
    """
    from scipy import integrate
    
    # Gauss-Laguerre quadrature: nodes and weights for int_0^inf f(x) e^{-x} dx
    # P(x)^2 = prod (x-r_i)^4, degree 4m=48. x*P(x)^2 has degree 49.
    # Need enough nodes to integrate a degree-49 polynomial exactly: need >= 50 nodes
    # But the function is not a polynomial times e^{-x} in the standard sense;
    # P(x)^2 * e^{-x} = [prod (x-r_i)^4] * e^{-x}
    # Gauss-Laguerre with n nodes integrates x^{n-1} e^{-x} exactly, so for
    # degree 49 we need n >= 50. Use more for safety.
    n_quad = max(100, 5 * m)
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Evaluate P(x)^2 at quadrature points
    vals = np.ones_like(x_quad)
    for r in roots:
        d = x_quad - r
        vals *= d * d  # (x-r)^2
    vals *= vals  # P(x)^2
    
    den = np.sum(w_quad * vals)
    num = np.sum(w_quad * vals * x_quad)
    
    if den <= 0 or not np.isfinite(den):
        return 1e10
    
    return num / den


def _objective(roots):
    """Compute the actual C4 upper bound. Lower is better."""
    roots = np.sort(roots)
    if roots.min() <= 0:
        return 1e10
    if np.any(np.diff(roots) < 1e-6):
        return 1e10
    return _bound_from_roots(roots)


def construct():
    """Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1). Score is the upper bound on C4 (lower is better)."""
    # Seed: [4, 8, 12, ..., 48]
    seed = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    best = seed.copy()
    best_val = _objective(seed)
    
    try:
        from scipy.optimize import minimize
        
        # Optimize in log-space to keep roots positive
        def log_objective(logs):
            r = np.exp(logs)
            r = np.sort(r)
            if r.min() <= 1e-6:
                return 1e10
            if np.any(np.diff(r) < 1e-4):
                return 1e10
            return _bound_from_roots(r)
        
        x0 = np.log(seed).astype(float)
        
        best_local = best.copy()
        best_local_val = best_val
        
        rng = np.random.default_rng(42)
        
        # Starting points: seed + perturbations
        starts = [x0.copy()]
        for _ in range(4):
            cand = x0 + rng.normal(0.0, 0.3, size=12)
            starts.append(cand)
        
        for s_idx, s in enumerate(starts):
            if _remaining() < 60.0:
                break
            
            try:
                res = minimize(
                    log_objective,
                    s,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 5000,
                        "xatol": 1e-6,
                        "fatol": 1e-8,
                        "adaptive": True,
                    },
                )
                if res.fun < best_local_val:
                    r = np.sort(np.exp(res.x))
                    if r.min() > 1e-4 and np.all(np.diff(r) > 1e-3):
                        best_local = r.copy()
                        best_local_val = float(res.fun)
            except Exception:
                pass
        
        if best_local_val < best_val:
            best = best_local.copy()
            best_val = best_local_val
    
    except Exception:
        pass
    
    # Fallback: try scaling the seed down
    if best_val >= 1.0:
        for scale in [0.5, 0.7, 0.8, 0.9]:
            cand = seed * scale
            val = _objective(cand)
            if val < best_val:
                best = cand.copy()
                best_val = val
    
    return [float(x) for x in np.sort(best)]
