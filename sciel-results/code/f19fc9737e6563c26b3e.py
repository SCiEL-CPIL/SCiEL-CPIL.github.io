"""Optimized 8-point configuration maximizing product of distances at fixed diameter."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import cos, sin, pi, log, exp

def _score(pts):
    """Compute Delta / n^n for 8 points, scaling to diameter 2."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    # diameter
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d[i, j] = d[j, i] = np.linalg.norm(pts[i] - pts[j])
    diam = d.max()
    if diam < 1e-15:
        return 0.0
    pts = pts / (diam / 2.0)  # scale to diameter 2
    # product of all ordered pair distances
    log_prod = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < 1e-300:
                    return 0.0
                log_prod += log(dist)
    score = exp(log_prod - n * log(n))
    return score

def _score_fast(pts):
    """Vectorized score computation."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, 0.0)
    diam = dist.max()
    if diam < 1e-15:
        return 0.0
    dist = dist / (diam / 2.0)
    # product over ordered pairs (off-diagonal)
    mask = ~np.eye(n, dtype=bool)
    vals = dist[mask]
    vals = vals[vals > 1e-300]
    if len(vals) == 0:
        return 0.0
    log_prod = np.sum(np.log(vals))
    score = exp(log_prod - n * log(n))
    return score

def construct():
    start = time.time()
    deadline = start + 140.0  # 10s safety margin
    
    # Start with regular octagon
    oct_pts = np.array([[cos(2*pi*k/8), sin(2*pi*k/8)] for k in range(8)])
    best_score = _score_fast(oct_pts)
    best_pts = oct_pts.copy()
    
    # Differential Evolution
    from scipy.optimize import differential_evolution
    
    def objective(x):
        pts = x.reshape(8, 2)
        s = _score_fast(pts)
        return -s  # minimize negative score
    
    # Bounds: keep coordinates reasonable
    bounds = [(-5.0, 5.0)] * 16
    
    # Run DE with time budget
    elapsed = time.time() - start
    if elapsed < deadline - 20:
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=500, popsize=15,
                tol=1e-10, seed=42,
                init='sobol',
                updating='deferred',
                polish=False
            )
            if result.success or result.fun < 0:
                cand = result.x.reshape(8, 2)
                s = _score_fast(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    elapsed = time.time() - start
    # Local polish with L-BFGS-B on log-objective
    if elapsed < deadline - 5:
        from scipy.optimize import minimize
        
        def neg_log_obj(x):
            pts = x.reshape(8, 2)
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dist, 0.0)
            diam = dist.max()
            if diam < 1e-15:
                return 1e10
            dist = dist / (diam / 2.0)
            mask = ~np.eye(8, dtype=bool)
            vals = dist[mask]
            vals = vals[vals > 1e-300]
            if len(vals) == 0:
                return 1e10
            return -(np.sum(np.log(vals)) - 8 * log(8))
        
        def neg_log_obj_grad(x):
            # Numerical gradient (fallback)
            pts = x.reshape(8, 2)
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dist, 0.0)
            diam = dist.max()
            if diam < 1e-15:
                return np.zeros(16)
            dist = dist / (diam / 2.0)
            mask = ~np.eye(8, dtype=bool)
            vals = dist[mask]
            vals = vals[vals > 1e-300]
            if len(vals) == 0:
                return np.zeros(16)
            # Use finite differences for gradient
            eps = 1e-7
            grad = np.zeros(16)
            f0 = neg_log_obj(x)
            for i in range(16):
                xp = x.copy()
                xp[i] += eps
                grad[i] = (neg_log_obj(xp) - f0) / eps
            return grad
        
        # Multiple restarts for local polish
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            x0 = best_pts.flatten().copy()
            if attempt == 1:
                # Perturb slightly
                x0 += np.random.default_rng(attempt * 123 + 7).normal(0, 0.1, 16)
            elif attempt == 2:
                # Larger perturbation
                x0 += np.random.default_rng(attempt * 456 + 13).normal(0, 0.5, 16)
            
            try:
                res = minimize(
                    neg_log_obj, x0,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
                )
                cand = res.x.reshape(8, 2)
                s = _score_fast(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
            except Exception:
                pass
    
    # Final: try a few more DE restarts if time permits
    elapsed = time.time() - start
    if elapsed < deadline - 15:
        for seed_val in [1, 2, 3]:
            if time.time() > deadline - 15:
                break
            try:
                result = differential_evolution(
                    objective, bounds,
                    maxiter=300, popsize=10,
                    tol=1e-10, seed=seed_val,
                    init='sobol',
                    updating='deferred',
                    polish=False
                )
                cand = result.x.reshape(8, 2)
                s = _score_fast(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
            except Exception:
                pass
    
    # Safety: check deadline
    if time.time() > deadline:
        pass  # just return best
    
    # Ensure distinct points and valid coordinates
    pts = best_pts.tolist()
    # Check distinctness
    for i in range(8):
        for j in range(i+1, 8):
            if abs(pts[i][0] - pts[j][0]) < 1e-12 and abs(pts[i][1] - pts[j][1]) < 1e-12:
                pts[j][0] += 1e-6
    
    return pts
# EVOLVE-BLOCK-END
